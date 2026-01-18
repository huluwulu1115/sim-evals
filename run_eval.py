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
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

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
        ):
    _torch_cuda_arch_preflight()
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


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    # Default instructions (can be overridden via --instruction)
    if instruction is None:
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
    
    if stage is not None:
        # Remove or hide other objects that aren't the cube
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

    # Create the inference client
    client = client.lower().strip()
    if client == "jointpos":
        client = DroidJointPosClient(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=open_loop_horizon,
            debug_vlm=print_vlm,
            vlm_head_dim=vlm_head_dim,
            vlm_full=vlm_full,
        )
    elif client == "jointvel":
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


    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            for step in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, instruction)
                if print_vlm and ret.get("vlm") is not None:
                    tqdm.write(f"[ep={ep} step={step}] {_format_vlm_debug(ret['vlm'])}")
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
