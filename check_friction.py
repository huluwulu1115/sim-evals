#!/usr/bin/env python3
"""Check physics material friction values for gripper and objects."""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Check friction values")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, UsdShade
from pathlib import Path

DATA_PATH = Path(__file__).parent / "assets"
OBJECTS_PATH = Path(__file__).parent / "objects"


def check_usd_friction(usd_path: str, name: str):
    """Check physics materials in a USD file."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"File: {usd_path}")
    
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"  ERROR: Could not open USD file")
        return
    
    # Find physics materials
    material_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat_api = UsdPhysics.MaterialAPI(prim)
            static_friction = mat_api.GetStaticFrictionAttr().Get()
            dynamic_friction = mat_api.GetDynamicFrictionAttr().Get()
            restitution = mat_api.GetRestitutionAttr().Get()
            print(f"\n  PhysicsMaterial: {prim.GetPath()}")
            print(f"    Static Friction:  {static_friction}")
            print(f"    Dynamic Friction: {dynamic_friction}")
            print(f"    Restitution:      {restitution}")
            material_count += 1
    
    if material_count == 0:
        print("\n  No PhysicsMaterialAPI found.")
    
    # Check collision prims for friction
    print("\n  Collision prims with friction properties:")
    collision_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
            name_lower = prim.GetName().lower()
            # Only show gripper-related or interesting prims
            if any(x in name_lower for x in ['finger', 'pad', 'gripper', 'tip', 'link']):
                # Check if there's a physics material binding
                binding_api = UsdShade.MaterialBindingAPI(prim)
                physics_mat = None
                if binding_api:
                    # Try to get physics material binding
                    for target in binding_api.GetDirectBindingRel().GetTargets():
                        mat_prim = stage.GetPrimAtPath(target)
                        if mat_prim and mat_prim.HasAPI(UsdPhysics.MaterialAPI):
                            physics_mat = target
                            break
                print(f"    {prim.GetPath()}")
                if physics_mat:
                    print(f"      -> Physics Material: {physics_mat}")
    
    print(f"\n  Total collision prims: {collision_count}")


def main():
    # Check robot gripper
    robot_usd = DATA_PATH / "franka_robotiq_2f_85_flattened.usd"
    check_usd_friction(robot_usd, "GRIPPER (Franka + Robotiq 2F-85)")
    
    # Check stapler objects
    for obj_dir in OBJECTS_PATH.iterdir():
        if obj_dir.is_dir():
            # Look for converted USD (from URDF conversion cache)
            # The URDF gets converted to USD at runtime, but we can check the URDF for friction
            urdf_path = obj_dir / "mobility_revised.urdf"
            if not urdf_path.exists():
                urdf_path = obj_dir / "mobility_human.urdf"
            if not urdf_path.exists():
                urdf_path = obj_dir / "mobility.urdf"
            
            if urdf_path.exists():
                print(f"\n{'=' * 60}")
                print(f"OBJECT: {obj_dir.name}")
                print(f"{'=' * 60}")
                print(f"URDF: {urdf_path}")
                print("\n  Note: URDF doesn't define contact friction directly.")
                print("  Contact friction is set during URDF->USD conversion.")
                print("  Default IsaacLab URDF conversion uses:")
                print("    Static Friction:  0.5 (default)")
                print("    Dynamic Friction: 0.5 (default)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Contact friction is determined by physics materials on collision geometry.

For the GRIPPER:
  - Check the USD file's PhysicsMaterialAPI properties
  - Robotiq 2F-85 gripper pads typically have high friction (~1.0)
  
For URDF OBJECTS (stapler, etc.):
  - IsaacLab's URDF converter applies default friction values
  - These can be overridden via collision_props in UrdfFileCfg
  - Default: static_friction=0.5, dynamic_friction=0.5

To modify contact friction, add to UrdfFileCfg:
  collision_props=sim_utils.CollisionPropertiesCfg(
      contact_offset=0.005,
      rest_offset=0.0,
  )
  
And/or apply a physics material after spawning.
""")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
