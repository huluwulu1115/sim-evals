#!/usr/bin/env python3
"""
Zero Agent for sim-evals DROID environment with URDF objects.

Runs the simulation with zero actions to verify the scene loads correctly
and inspect object properties (joint positions, dynamics, physics).

Usage:
    # Basic usage - load an object and run simulation
    python run_zero_agent.py --object 103275_evogen

    # With object inspection
    python run_zero_agent.py --object 103275_evogen --inspect

    # Headless mode for testing
    python run_zero_agent.py --object 103275_evogen --headless --steps 500

    # Available objects:
    #   103275_evogen, 103275_ivw, 103275_humangensim2, 103275_vlm1shot
"""

import tyro
import argparse
import gymnasium as gym
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


def _torch_cuda_arch_preflight() -> None:
    """Fail fast with a clear message when installed torch can't run on GPU."""
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
                f"Your GPU ({name}) has CUDA arch {arch}, but this torch only supports: {sorted(supported)}.\n"
                "Install a newer torch wheel that supports your GPU."
            )
    except Exception as e:
        raise RuntimeError(str(e)) from e


def _load_grasping_task_config(object_path: str):
    """Load grasping task config from objects/<category>/config.py.

    Returns:
        (object_config, grasping_config) or (None, None) on failure.
    """
    import importlib

    parts = object_path.split("/")
    if len(parts) != 2:
        print(f"[CONFIG] Object path must be category/variant format: {object_path}")
        return None, None

    category, variant = parts
    try:
        config_module = importlib.import_module(f"objects.{category}.config")
        object_config = config_module.get_config(variant)
        return object_config, object_config.grasping
    except ImportError as e:
        print(f"[CONFIG] Could not import config for {category}: {e}")
        return None, None
    except ValueError as e:
        print(f"[CONFIG] {e}")
        return None, None


def print_object_properties(env):
    """Print detailed object properties from the environment."""
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env.env
        
        print(f"\n{'='*70}")
        print("OBJECT PROPERTIES")
        print(f"{'='*70}")
        
        # Check for articulation object
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'articulations'):
            for name, obj in unwrapped.scene.articulations.items():
                if name in ['robot', 'bowl']:
                    continue
                print(f"\nArticulation: {name}")
                print(f"  Type: {type(obj).__name__}")
                
                if hasattr(obj, 'data'):
                    data = obj.data
                    
                    # Root pose
                    if hasattr(data, 'root_pos_w'):
                        pos = data.root_pos_w[0].cpu().numpy()
                        print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                    
                    if hasattr(data, 'root_quat_w'):
                        quat = data.root_quat_w[0].cpu().numpy()
                        print(f"  Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                    
                    # Joints
                    if hasattr(data, 'joint_names') and data.joint_names:
                        print(f"\n  Joints ({len(data.joint_names)}):")
                        for i, jname in enumerate(data.joint_names):
                            print(f"    [{i}] {jname}")
                            
                            if hasattr(data, 'joint_pos') and data.joint_pos.numel() > 0:
                                pos = data.joint_pos[0, i].item()
                                print(f"        Position: {pos:.4f} rad ({np.degrees(pos):.2f}°)")
                            
                            if hasattr(data, 'soft_joint_pos_limits'):
                                limits = data.soft_joint_pos_limits[0, i].cpu().numpy()
                                print(f"        Limits: [{limits[0]:.4f}, {limits[1]:.4f}] rad")
                            
                            if hasattr(data, 'joint_stiffness'):
                                stiff = data.joint_stiffness[0, i].item()
                                print(f"        Stiffness: {stiff:.4f}")
                            
                            if hasattr(data, 'joint_damping'):
                                damp = data.joint_damping[0, i].item()
                                print(f"        Damping: {damp:.4f}")
                            
                            if hasattr(data, 'joint_friction_coeff'):
                                fric = data.joint_friction_coeff[0, i].item()
                                print(f"        Friction: {fric:.4f}")
        
        # Check rigid objects
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            for name, obj in unwrapped.scene.rigid_objects.items():
                if name == 'table':
                    continue
                print(f"\nRigid Object: {name}")
                if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                    pos = obj.data.root_pos_w[0].cpu().numpy()
                    print(f"  Position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"[WARNING] Error printing object properties: {e}")


def main(
    object: str | None = None,
    object_pos: tuple[float, float, float] | None = None,
    object_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    fix_base: bool = False,
    keep_bowl: bool = False,
    bowl_pos: tuple[float, float, float] | None = None,
    headless: bool = False,
    steps: int = 0,
    inspect: bool = False,
    scene: int = 1,
):
    """Run zero agent simulation for sim-evals DROID environment.
    
    Args:
        object: Object folder name in sim-evals/objects/ (e.g., '103275_evogen')
        object_pos: Object position (x, y, z). If not set, auto-computed from bounding_box.json
        object_rot: Object rotation quaternion (w, x, y, z)
        fix_base: If True, fix the object's base in place (no physics on root)
        keep_bowl: If True, add bowl to the scene
        bowl_pos: Bowl position (x, y, z). Default: (0.5, 0.15, 0.0)
        headless: Run without GUI
        steps: Number of steps to run (0 = run until closed)
        inspect: Print detailed object properties
        scene: Scene number if not using --object mode
    """
    _torch_cuda_arch_preflight()
    
    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Zero agent for sim-evals.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import after app launch
    import src.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # If an object is provided and CLI didn't override, pull defaults from objects/<category>/config.py
    if object is not None:
        object_config, grasping_config = _load_grasping_task_config(object)

        # Rotation
        if object_config is not None and object_rot == (1.0, 0.0, 0.0, 0.0):
            object_rot = object_config.initial_rotation
            print(f"[CONFIG] Using initial_rotation from config: {object_rot}")

        # Positions and bowl settings (grasping layout)
        if grasping_config is not None:
            if object_pos is None and getattr(grasping_config, "object_pos", None) is not None:
                object_pos = grasping_config.object_pos
                print(f"[CONFIG] Using object_pos from config: {object_pos}")
            # Use keep_bowl from config if not already set via CLI
            if not keep_bowl and getattr(grasping_config, "keep_bowl", False):
                keep_bowl = True
                print(f"[CONFIG] keep_bowl=True from config")
            if keep_bowl and bowl_pos is None and getattr(grasping_config, "bowl_pos", None) is not None:
                bowl_pos = grasping_config.bowl_pos
                print(f"[CONFIG] Using bowl_pos from config: {bowl_pos}")

    print(f"\n{'='*70}")
    print("ZERO AGENT - SIM-EVALS DROID ENVIRONMENT")
    print(f"{'='*70}")
    if object:
        print(f"Object: {object}")
        print(f"Position: {object_pos if object_pos else 'auto (from bounding_box.json)'}")
        print(f"Rotation: {object_rot}")
    else:
        print(f"Scene: {scene}")
    print(f"Headless: {headless}")
    print(f"Steps: {steps if steps > 0 else 'unlimited'}")
    print(f"{'='*70}\n")

    # Initialize environment
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    # Configure scene
    if object is not None:
        print(f"[SCENE] Loading URDF object: {object}")
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
        env_cfg.set_scene(scene)

    env = gym.make("DROID", cfg=env_cfg)
    obs, _ = env.reset()
    obs, _ = env.reset()  # Second reset for materials

    # Print object properties if requested
    if inspect:
        print_object_properties(env)

    # Run simulation
    print("[INFO] Starting zero-action simulation...")
    print("[INFO] Press Ctrl+C or close window to exit.\n")

    step_count = 0
    max_steps = env.env.max_episode_length if steps == 0 else steps
    
    with torch.no_grad():
        while simulation_app.is_running():
            # Zero actions
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, term, trunc, info = env.step(action)
            step_count += 1

            # Print periodic status
            if step_count % 100 == 0:
                try:
                    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env.env
                    if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'articulations'):
                        for name, obj in unwrapped.scene.articulations.items():
                            if name == 'robot':
                                continue
                            if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                                pos = obj.data.root_pos_w[0].cpu().numpy()
                                msg = f"Step {step_count}: Object pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                                if hasattr(obj.data, 'joint_pos') and obj.data.joint_pos.numel() > 0:
                                    jp = obj.data.joint_pos[0].cpu().numpy()
                                    msg += f", joints={[f'{j:.3f}' for j in jp]}"
                                print(msg)
                except Exception:
                    print(f"Step {step_count}")

            # Check termination
            if steps > 0 and step_count >= steps:
                print(f"\n[INFO] Completed {steps} steps.")
                break

            if term or trunc:
                if steps == 0:
                    obs, _ = env.reset()
                    print(f"[INFO] Episode reset at step {step_count}")

    env.close()
    simulation_app.close()
    print("[INFO] Simulation closed.")


if __name__ == "__main__":
    args = tyro.cli(main)
