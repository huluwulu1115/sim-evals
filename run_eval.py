"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless

# With task config (grasping or tool_use):
python run_eval.py --object stapler/103275_evogen --task grasping --episodes 10 --headless
python run_eval.py --object stapler/103275_evogen --task tool_use --episodes 10 --headless
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Literal

from src.inference.droid_jointpos import Client as DroidJointPosClient
from src.inference.droid_jointvel import Client as DroidJointVelClient


def _torch_cuda_arch_preflight() -> None:
    """Fail fast with a clear message when the installed torch can't run on the GPU.

    New GPUs (e.g. RTX 5090 / sm_120) require a torch build that includes kernels/PTX for that arch.
    If not, IsaacLab will crash with: "CUDA error: no kernel image is available for execution on the device".
    """
    import torch

    if not torch.cuda.is_available():
        return
    try:
        cap = torch.cuda.get_device_capability(0)
        arch = f"sm_{cap[0]}{cap[1]}"
        supported = set(torch.cuda.get_arch_list())
        if arch not in supported:
            name = torch.cuda.get_device_name(0)
            raise RuntimeError(
                f"Your GPU ({name}) has CUDA arch {arch}, but this torch build only supports: {sorted(supported)}.\n"
                "Fix: install a newer torch wheel that supports your GPU (often nightly), OR run eval on CPU.\n\n"
                "Option 1 - Install into conda environment (if using conda):\n"
                "  conda activate env_isaaclab_5.x\n"
                "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128\n\n"
                "Option 2 - Install into the *sim-evals* uv environment:\n"
                "  uv run python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128\n\n"
                "Nightly (if stable still doesn't include sm_120):\n"
                "  pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128\n\n"
                "CPU fallback:\n"
                "  python run_eval.py ... --device cpu\n"
            )
    except Exception as e:
        # Re-raise as RuntimeError so it's visible and stops early.
        raise RuntimeError(str(e)) from e


def _format_vlm_debug(vlm: dict) -> str:
    """Format the VLM debug payload returned by openpi into a single, readable line."""
    if not isinstance(vlm, dict):
        return f"VLM_DEBUG (unexpected type): {type(vlm)}"
    if "error" in vlm:
        return f"VLM_DEBUG error: {vlm.get('error')}"

    parts: list[str] = []
    parts.append(f"pi05={vlm.get('pi05', None)}")
    if "prefix_tokens" in vlm:
        parts.append(f"prefix_tokens={vlm['prefix_tokens']}")
    if "prompt_tokens" in vlm:
        parts.append(f"prompt_tokens={vlm['prompt_tokens']}")

    def _vec_summary(name: str, obj: dict | None) -> None:
        if not isinstance(obj, dict):
            return
        norm = obj.get("norm", None)
        head = obj.get("head", None)
        if isinstance(norm, (int, float)):
            parts.append(f"{name}_norm={norm:.3f}")
        if head is not None:
            head_arr = np.asarray(head)
            # Keep it compact even if head_dim is large.
            parts.append(
                f"{name}_head={np.array2string(head_arr, precision=3, floatmode='fixed', separator=',', max_line_width=120)}"
            )

    _vec_summary("prefix", vlm.get("prefix"))
    if "prompt" in vlm:
        _vec_summary("prompt", vlm.get("prompt"))

    images = vlm.get("images")
    if isinstance(images, dict):
        img_parts: list[str] = []
        for k, v in images.items():
            if not isinstance(v, dict):
                continue
            norm = v.get("norm", None)
            mask = v.get("mask", None)
            if isinstance(norm, (int, float)) and isinstance(mask, bool):
                img_parts.append(f"{k}(mask={mask},norm={norm:.3f})")
            elif isinstance(norm, (int, float)):
                img_parts.append(f"{k}(norm={norm:.3f})")
        if img_parts:
            parts.append("images=" + " ".join(img_parts))

    return "VLM_DEBUG " + " ".join(parts)


def _load_task_config(object_path: str, task: str | None = None):
    """Load task configuration from objects/<category>/config.py.
    
    Args:
        object_path: Object path like "stapler/103275_evogen"
        task: Task type - "grasping" or "tool_use" (optional)
    
    Returns:
        Tuple of (object_config, task_config, instruction) or (None, None, None) if not found
    """
    import importlib
    
    parts = object_path.split("/")
    if len(parts) != 2:
        print(f"[CONFIG] Object path must be category/variant format: {object_path}")
        return None, None, None
    
    category, variant = parts
    
    try:
        # Import the category's config module
        config_module = importlib.import_module(f"objects.{category}.config")
        object_config = config_module.get_config(variant)
        
        if task is None:
            # Just return object config without task config
            return object_config, None, None
        
        if task == "grasping" and object_config.grasping:
            return object_config, object_config.grasping, object_config.grasping.prompt
        elif task == "tool_use" and object_config.tool_use:
            return object_config, object_config.tool_use, object_config.tool_use.prompt
        else:
            print(f"[CONFIG] Task '{task}' not configured for {object_path}")
            return object_config, None, None
            
    except ImportError as e:
        print(f"[CONFIG] Could not import config for {category}: {e}")
        return None, None, None
    except ValueError as e:
        print(f"[CONFIG] {e}")
        return None, None, None


def _get_object_state(env, object_name: str = "object"):
    """Get object position and joint positions from environment.
    
    Returns:
        Tuple of (root_pos_z, joint_positions_dict) or (None, None) if not found
    """
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env.env
        
        # Check articulations (for URDF objects)
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'articulations'):
            for name, obj in unwrapped.scene.articulations.items():
                if name in ['robot', 'bowl']:
                    continue
                if hasattr(obj, 'data'):
                    data = obj.data
                    root_z = None
                    joint_positions = {}
                    
                    if hasattr(data, 'root_pos_w'):
                        root_z = data.root_pos_w[0, 2].item()
                    
                    if hasattr(data, 'joint_names') and hasattr(data, 'joint_pos'):
                        for i, jname in enumerate(data.joint_names):
                            joint_positions[jname] = data.joint_pos[0, i].item()
                    
                    return root_z, joint_positions
        
        # Check rigid objects (for simple objects like cube)
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            for name, obj in unwrapped.scene.rigid_objects.items():
                if name in ['table']:
                    continue
                if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                    root_z = obj.data.root_pos_w[0, 2].item()
                    return root_z, {}
                    
    except Exception as e:
        print(f"[WARNING] Error getting object state: {e}")
    
    return None, None


def _print_object_info(env):
    """Print detailed object physics properties (joints, masses, etc.)."""
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env.env
        
        print(f"\n{'='*60}")
        print("OBJECT PHYSICS PROPERTIES")
        print(f"{'='*60}")
        
        # Check articulations (for URDF objects)
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'articulations'):
            for name, obj in unwrapped.scene.articulations.items():
                if name in ['robot', 'bowl']:
                    continue
                    
                print(f"\nArticulation: {name}")
                print(f"{'-'*40}")
                
                # Print joint information
                if hasattr(obj, 'data'):
                    data = obj.data
                    
                    # Joint names and positions
                    if hasattr(data, 'joint_names'):
                        print(f"\nJoints ({len(data.joint_names)}):")
                        for i, jname in enumerate(data.joint_names):
                            pos = data.joint_pos[0, i].item() if hasattr(data, 'joint_pos') else 'N/A'
                            print(f"  [{i}] {jname}: pos={pos:.4f}" if isinstance(pos, float) else f"  [{i}] {jname}: pos={pos}")
                    
                    # Joint stiffness and damping
                    if hasattr(data, 'joint_stiffness'):
                        stiffness = data.joint_stiffness[0].cpu().numpy()
                        print(f"\nJoint Stiffness:")
                        for i, jname in enumerate(data.joint_names):
                            print(f"  {jname}: {stiffness[i]:.4f}")
                    
                    if hasattr(data, 'joint_damping'):
                        damping = data.joint_damping[0].cpu().numpy()
                        print(f"\nJoint Damping:")
                        for i, jname in enumerate(data.joint_names):
                            print(f"  {jname}: {damping[i]:.4f}")
                    
                    # Joint limits
                    if hasattr(data, 'joint_limits'):
                        limits = data.joint_limits[0].cpu().numpy()
                        print(f"\nJoint Limits (lower, upper):")
                        for i, jname in enumerate(data.joint_names):
                            print(f"  {jname}: [{limits[i, 0]:.4f}, {limits[i, 1]:.4f}]")
                
                # Print body/link masses
                if hasattr(obj, 'root_physx_view') and obj.root_physx_view is not None:
                    try:
                        masses = obj.root_physx_view.get_masses()
                        if masses is not None:
                            masses = masses[0].cpu().numpy()
                            print(f"\nLink Masses:")
                            body_names = data.body_names if hasattr(data, 'body_names') else [f"body_{i}" for i in range(len(masses))]
                            total_mass = 0.0
                            for i, bname in enumerate(body_names):
                                print(f"  {bname}: {masses[i]:.6f} kg")
                                total_mass += masses[i]
                            print(f"  TOTAL: {total_mass:.6f} kg")
                    except Exception as e:
                        print(f"  [Could not get masses: {e}]")
        
        # Check rigid objects
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            for name, obj in unwrapped.scene.rigid_objects.items():
                if name in ['table']:
                    continue
                print(f"\nRigid Object: {name}")
                print(f"{'-'*40}")
                
                if hasattr(obj, 'root_physx_view') and obj.root_physx_view is not None:
                    try:
                        masses = obj.root_physx_view.get_masses()
                        if masses is not None:
                            print(f"  Mass: {masses[0].item():.6f} kg")
                    except Exception as e:
                        print(f"  [Could not get mass: {e}]")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"[WARNING] Error printing object info: {e}")


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 1,
        instruction: str | None = None,
        client: str = "jointvel",
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        print_vlm: bool = False,
        vlm_head_dim: int = 16,
        vlm_full: bool = False,
        # Optional: replace an object prim inside the selected USD scene.
        object_usd: str | None = None,
        object_prim: str | None = None,
        # Load a URDF object from sim-evals/objects/ folder (table + object only)
        object: str | None = None,
        object_pos: tuple[float, float, float] | None = None,
        object_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        fix_base: bool = False,
        no_auto_fix_base: bool = False,  # Disable automatic fix_base for tool_use tasks
        keep_bowl: bool = False,
        bowl_pos: tuple[float, float, float] | None = None,
        # Task configuration
        task: Literal["grasping", "tool_use"] | None = None,
        ):
    _torch_cuda_arch_preflight()
    
    # Load object and task config if specified
    object_config = None
    task_config = None
    if object is not None:
        object_config, task_config, config_instruction = _load_task_config(object, task)
        if object_config is not None:
            # Use initial_rotation from config if not overridden by CLI
            if object_rot == (1.0, 0.0, 0.0, 0.0):  # Default value
                object_rot = object_config.initial_rotation
                print(f"[CONFIG] Using initial_rotation from config: {object_rot}")
        if task_config is not None and instruction is None:
            instruction = config_instruction
            print(f"[CONFIG] Loaded {task} config for {object}")
            print(f"[CONFIG] Instruction: {instruction}")
        
        # For grasping tasks, use scene layout from config
        if task == "grasping" and task_config is not None:
            # Use keep_bowl from config (default False if not specified)
            if hasattr(task_config, 'keep_bowl') and task_config.keep_bowl:
                keep_bowl = True
                print(f"[CONFIG] keep_bowl=True from config")
            # Use object_pos from config if not overridden by CLI (handle z=None)
            if object_pos is None and hasattr(task_config, 'object_pos') and task_config.object_pos is not None:
                cfg_pos = task_config.object_pos
                # If z is None, let it be computed from bounding box (pass x, y only)
                if cfg_pos[2] is None:
                    object_pos = (cfg_pos[0], cfg_pos[1], None)
                else:
                    object_pos = cfg_pos
                print(f"[CONFIG] Using object_pos from config: {object_pos}")
            # Use bowl_pos from config if not overridden by CLI (only if keep_bowl is True)
            if keep_bowl and bowl_pos is None and hasattr(task_config, 'bowl_pos') and task_config.bowl_pos is not None:
                bowl_pos = task_config.bowl_pos
                print(f"[CONFIG] Using bowl_pos from config: {bowl_pos}")
    
    # For tool_use tasks, use config values
    if task == "tool_use" and task_config is not None:
        # Use object_pos from config if not overridden by CLI
        if object_pos is None and hasattr(task_config, 'object_pos') and task_config.object_pos is not None:
            cfg_pos = task_config.object_pos
            if cfg_pos[2] is None:
                object_pos = (cfg_pos[0], cfg_pos[1], None)
            else:
                object_pos = cfg_pos
            print(f"[CONFIG] Using object_pos from config: {object_pos}")
        
        # Use fix_base from config (defaults to True)
        # CLI --fix_base or --no_auto_fix_base can override
        if not no_auto_fix_base:
            if hasattr(task_config, 'fix_base'):
                config_fix_base = task_config.fix_base
                if not fix_base:  # CLI didn't explicitly set fix_base
                    fix_base = config_fix_base
                    print(f"[CONFIG] Using fix_base={fix_base} from config")
            elif not fix_base:
                fix_base = True
                print(f"[CONFIG] Automatically enabling fix_base=True for tool_use task")
    elif task == "tool_use" and not no_auto_fix_base and not fix_base:
        fix_base = True
        print(f"[CONFIG] Automatically enabling fix_base=True for tool_use task")
    
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    
    # Import success check functions
    from objects import check_grasping_success, check_tool_use_success


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    # Default instructions (can be overridden via --instruction)
    if instruction is None:
        if object is not None:
            # For URDF object mode, use a generic instruction
            instruction = "interact with the object on the table"
        else:
            match scene:
                case 1:
                    instruction = "put the cube in the bowl"
                case 2:
                    instruction = "put the can in the mug"
                case 3:
                    instruction = "put banana in the bin"
                case _:
                    # Allow custom scenes without a baked-in instruction.
                    instruction = "do something"

    # Configure scene based on mode: URDF object or USD scene
    if object is not None:
        # Load scene USD (table/background) + URDF object
        print(f"[SCENE] Loading URDF object mode: {object}")
        env_cfg.set_scene_with_object(
            object_name=object,
            scene_name=str(scene),
            object_pos=object_pos,
            object_rot=object_rot,
            fix_base=fix_base,
            keep_bowl=keep_bowl,
            bowl_pos=bowl_pos,
        )
    else:
        # Original USD scene mode
        # Optional object replacement inside the scene USD.
        if object_usd is not None:
            # Reasonable defaults for prim names based on shipped scenes.
            if object_prim is None:
                match scene:
                    case 1:
                        object_prim = "cube"
                    case 2:
                        object_prim = "can"
                    case 3:
                        object_prim = "banana"
                    case _:
                        raise ValueError("--object_prim is required when using --object_usd for unknown scenes.")
            env_cfg.scene.replace_object_usd_path = object_usd
            env_cfg.scene.replace_object_prim = object_prim
            
        env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    
    # Apply red material to cube (similar to franka_droid_sim2real_env.py)
    from src.environments.droid_environment import _apply_red_material_to_cube
    # Access stage through the environment
    stage = None
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'scene') and hasattr(env.unwrapped.scene, 'sim'):
        stage = env.unwrapped.scene.sim.stage
    elif hasattr(env, 'env') and hasattr(env.env, 'scene') and hasattr(env.env.scene, 'sim'):
        stage = env.env.scene.sim.stage
    
    if stage is not None and object is None:
        # Only apply these modifications for original USD scene mode (not URDF object mode).
        # In URDF object mode, the scene is already patched via _make_scene_without_objects.
        from pxr import UsdGeom, UsdPhysics
        
        def hide_non_cube_objects(root_prim_path: str):
            """Hide all rigid body objects that aren't cube/block/object."""
            root_prim = stage.GetPrimAtPath(root_prim_path)
            if not root_prim or not root_prim.IsValid():
                return
            for child in root_prim.GetChildren():
                if not UsdPhysics.RigidBodyAPI(child):
                    continue
                name = child.GetName().lower()
                # Keep only cube/block/object, hide everything else
                if name not in ["cube", "block", "object"]:
                    # Make invisible instead of deleting (safer)
                    imageable = UsdGeom.Imageable(child)
                    if imageable:
                        imageable.MakeInvisible()
                        print(f"Hidden object: {child.GetName()} at {child.GetPath()}")
        
        # Check both scene and direct env paths
        hide_non_cube_objects("/World/envs/env_0/scene")
        hide_non_cube_objects("/World/envs/env_0")
        
        # Try both possible cube paths (from scene or spawned)
        cube_paths = [
            "/World/envs/env_0/cube",
            "/World/envs/env_0/scene/cube",
        ]
        for cube_path in cube_paths:
            _apply_red_material_to_cube(stage, cube_path)
    
    # Apply object material properties from overlay (friction, restitution)
    if stage is not None and object is not None:
        from src.environments.droid_environment import apply_physics_material, _OBJECT_MATERIAL_PROPS
        
        # Get material properties from module-level storage
        friction = _OBJECT_MATERIAL_PROPS.get('friction', 0.5)
        restitution = _OBJECT_MATERIAL_PROPS.get('restitution', 0.1)
        
        # Apply to the object prim
        object_prim_path = "/World/envs/env_0/Object"
        apply_physics_material(
            stage,
            object_prim_path,
            static_friction=friction,
            dynamic_friction=friction,
            restitution=restitution,
        )

    # Create the inference client
    client_type = client.lower().strip()
    if client_type == "jointpos":
        client = DroidJointPosClient(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=open_loop_horizon,
            debug_vlm=print_vlm,
            vlm_head_dim=vlm_head_dim,
            vlm_full=vlm_full,
        )
    elif client_type == "jointvel":
        # dt in this env is ~15Hz (see droid_environment EnvCfg); keep it explicit here.
        client = DroidJointVelClient(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=open_loop_horizon,
            dt=1.0 / 15.0,
            debug_vlm=print_vlm,
            vlm_head_dim=vlm_head_dim,
            vlm_full=vlm_full,
        )
    else:
        raise ValueError("--client must be one of: jointpos, jointvel")


    # Build run name: [task]_[objecttype]_[objectid]_[method]
    if object is not None:
        # Parse object path like "stapler/103275_evogen" -> objecttype="stapler", objectid="103275", method="evogen"
        parts = object.split("/")
        if len(parts) == 2:
            objecttype = parts[0]  # e.g., "stapler"
            variant = parts[1]  # e.g., "103275_evogen"
            variant_parts = variant.split("_", 1)
            objectid = variant_parts[0]  # e.g., "103275"
            method = variant_parts[1] if len(variant_parts) > 1 else "unknown"  # e.g., "evogen"
        else:
            objecttype = "unknown"
            objectid = object
            method = "unknown"
        task_name = task if task else "notask"
        run_name = f"{task_name}_{objecttype}_{objectid}_{method}"
    else:
        run_name = f"scene{scene}"
    
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / run_name
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    recording_video = []  # Video from recording_cam
    ep = 0
    max_steps = env.env.max_episode_length
    
    # Evaluation results
    eval_results = {
        "object": object,
        "task": task,
        "instruction": instruction,
        "episodes": episodes,
        "timestamp": datetime.now().isoformat(),
        "results": [],
    }
    
    # Warm-up step to ensure cameras have valid data
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs, _, _, _, _ = env.step(zero_action)
    
    # Debug: Check observation structure
    print(f"[DEBUG] Observation keys: {obs.keys()}")
    if "policy" in obs:
        print(f"[DEBUG] Policy keys: {obs['policy'].keys()}")
        for key in ["external_cam", "wrist_cam", "recording_cam"]:
            if key in obs["policy"]:
                val = obs["policy"][key]
                print(f"[DEBUG] {key} shape: {val.shape if hasattr(val, 'shape') else type(val)}")
            else:
                print(f"[DEBUG] {key}: NOT FOUND")
    
    # Print object physics properties (joints, masses, etc.)
    if object is not None:
        _print_object_info(env)
    
    with torch.no_grad():
        for ep in range(episodes):
            # Reset environment at start of each episode
            obs, _ = env.reset()
            obs, _ = env.reset()  # Second reset for materials
            # Warm-up step for camera data
            obs, _, _, _, _ = env.step(zero_action)
            
            # Get initial object state for success checking
            initial_z, _ = _get_object_state(env)
            episode_success = False
            success_times = []  # List of times (in seconds) when success was achieved
            was_successful = False  # Track previous state for transition detection
            
            # Track best attempt metrics
            max_lift_height = 0.0  # For grasping: highest lift
            min_joint_error = float('inf')  # For tool use: closest to target
            best_joint_position = None  # The joint position at min error
            
            # Environment runs at 15Hz
            env_dt = 1.0 / 15.0
            
            for step in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if print_vlm and ret.get("vlm") is not None:
                    tqdm.write(f"[ep={ep} step={step}] {_format_vlm_debug(ret['vlm'])}")
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                
                # Capture recording_cam frame
                if "policy" in obs and "recording_cam" in obs["policy"]:
                    rec_frame = obs["policy"]["recording_cam"][0].cpu().numpy()
                    # Convert from (H, W, C) uint8 to proper format
                    if rec_frame.dtype != np.uint8:
                        rec_frame = (rec_frame * 255).astype(np.uint8)
                    recording_video.append(rec_frame)
                
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                
                # Check success criteria if task config is loaded
                if task_config is not None:
                    current_z, joint_positions = _get_object_state(env)
                    current_success = False
                    
                    if task == "grasping" and current_z is not None and initial_z is not None:
                        lift_height = current_z - initial_z
                        # Track max lift height
                        if lift_height > max_lift_height:
                            max_lift_height = lift_height
                        
                        current_success = check_grasping_success(initial_z, current_z, task_config)
                        if current_success and not was_successful:
                            # Transition from not successful to successful
                            time_seconds = step * env_dt
                            success_times.append(time_seconds)
                            if not episode_success:
                                episode_success = True
                            tqdm.write(f"[SUCCESS] Episode {ep+1}: Grasping success at step {step} ({time_seconds:.2f}s) (lifted {lift_height*100:.1f}cm)")
                    
                    elif task == "tool_use" and joint_positions:
                        # Track closest joint position to target
                        if task_config.target_joint in joint_positions:
                            current_pos = joint_positions[task_config.target_joint]
                            error = abs(current_pos - task_config.target_position)
                            if error < min_joint_error:
                                min_joint_error = error
                                best_joint_position = current_pos
                        
                        current_success = check_tool_use_success(joint_positions, task_config)
                        if current_success and not was_successful:
                            # Transition from not successful to successful
                            time_seconds = step * env_dt
                            success_times.append(time_seconds)
                            if not episode_success:
                                episode_success = True
                            tqdm.write(f"[SUCCESS] Episode {ep+1}: Tool use success at step {step} ({time_seconds:.2f}s)")
                    
                    was_successful = current_success
                
                if term or trunc:
                    break
            
            # Record episode result with best attempt metrics
            episode_result = {
                "episode": ep,
                "success": episode_success,
                "success_times": success_times,  # List of times (seconds) when success was achieved
                "first_success_time": success_times[0] if success_times else None,
                "success_count": len(success_times),
                "total_steps": step + 1,
            }
            
            # Add task-specific best attempt metrics
            if task == "grasping":
                episode_result["max_lift_height"] = max_lift_height
                episode_result["max_lift_height_cm"] = max_lift_height * 100
            elif task == "tool_use" and best_joint_position is not None:
                episode_result["best_joint_position"] = best_joint_position
                episode_result["min_error_to_target"] = min_joint_error
                episode_result["target_position"] = task_config.target_position
            
            eval_results["results"].append(episode_result)
            
            if task_config is not None:
                if episode_success:
                    status = "✓ SUCCESS"
                else:
                    if task == "grasping":
                        status = f"✗ FAILED (best lift: {max_lift_height*100:.1f}cm)"
                    elif task == "tool_use" and best_joint_position is not None:
                        status = f"✗ FAILED (best pos: {best_joint_position:.3f}, error: {min_joint_error:.3f})"
                    else:
                        status = "✗ FAILED"
                print(f"[EVAL] Episode {ep+1}/{episodes}: {status}")

            client.reset()
            
            # Save main video
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []
            
            # Save recording_cam video
            if recording_video:
                mediapy.write_video(
                    video_dir / f"episode_{ep}_recording.mp4",
                    recording_video,
                    fps=15,
                )
                recording_video = []
            
            # Save eval results after each episode (in case of crash)
            eval_file = video_dir / "eval.json"
            with open(eval_file, "w") as f:
                json.dump(eval_results, f, indent=2)
    
    # Calculate and save final metrics
    if task_config is not None:
        successes = sum(1 for r in eval_results["results"] if r["success"])
        success_rate = successes / episodes
        eval_results["success_rate"] = success_rate
        eval_results["successes"] = successes
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Object: {object}")
        print(f"Task: {task}")
        print(f"Instruction: {instruction}")
        print(f"Success Rate: {successes}/{episodes} ({success_rate*100:.1f}%)")
        print(f"{'='*50}\n")
    
    # Save evaluation results to JSON
    eval_file = video_dir / "eval.json"
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[EVAL] Results saved to {eval_file}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
