import json
import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np
import re
import xml.etree.ElementTree as ET

import hashlib
from typing import List, Dict
from pathlib import Path
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files import UrdfFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.sim import SimulationCfg
from pxr import Gf, Sdf, UsdShade

from .nvidia_droid import NVIDIA_DROID


# ============================================================================
# Friction Configuration
# ============================================================================


DEFAULT_STATIC_FRICTION = 0.5
DEFAULT_DYNAMIC_FRICTION = 0.5

_OBJECT_MATERIAL_PROPS: dict = {"friction": 0.5, "restitution": 0.1}

# Store initial joint positions for the object articulation
# This is set during scene configuration and used by the reset event
_OBJECT_INITIAL_JOINT_POS: dict = {}


def reset_object_joints_to_initial(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Custom reset event that writes object joint positions from URDF comments.
    
    This is needed because InitialStateCfg.joint_pos doesn't work reliably for
    URDF-loaded articulations. The task environments (DirectRLEnv) solve this by
    explicitly calling write_joint_state_to_sim in _reset_idx.
    
    For Manager-based environments, we use this custom event function.
    """
    global _OBJECT_INITIAL_JOINT_POS
    
    if not _OBJECT_INITIAL_JOINT_POS:
        return  # No initial positions configured
    
    # Get the object articulation
    try:
        obj = env.scene[asset_cfg.name]
    except KeyError:
        return  # Object not in scene (e.g., using cube instead of URDF object)
    
    # Check if it's an articulation with joints
    from isaaclab.assets import Articulation
    if not isinstance(obj, Articulation) or obj.num_joints == 0:
        return
    
    n = env_ids.shape[0]
    n_joints = obj.num_joints
    joint_pos_tensor = torch.zeros((n, n_joints), device=env.device)
    
    # Get joint names from the articulation
    joint_names = obj.data.joint_names
    
    # Map joint names to positions from stored initial positions
    for i, joint_name in enumerate(joint_names):
        if joint_name in _OBJECT_INITIAL_JOINT_POS:
            # Use value from URDF comments (already in radians)
            joint_pos_tensor[:, i] = _OBJECT_INITIAL_JOINT_POS[joint_name]
        else:
            # Fallback to default joint position from simulation
            joint_pos_tensor[:, i] = obj.data.default_joint_pos[0, i]
    
    # Write joint positions to simulation (this is the reliable way)
    joint_vel_tensor = torch.zeros_like(joint_pos_tensor)
    obj.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor, env_ids=env_ids)
    
    # Also set position target to ensure USD visual updates
    obj.set_joint_position_target(joint_pos_tensor, env_ids=env_ids)


def apply_physics_material(
    stage,
    prim_path: str,
    static_friction: float = DEFAULT_STATIC_FRICTION,
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION,
    restitution: float = 0.0,
):
    """Apply a physics material with custom friction to a prim.
    
    Args:
        stage: USD stage
        prim_path: Path to the prim (e.g., "/World/envs/env_0/Object")
        static_friction: Static friction coefficient
        dynamic_friction: Dynamic friction coefficient
        restitution: Bounciness (0 = no bounce, 1 = perfect bounce)
    """
    # Create physics material
    mat_path = f"{prim_path}/PhysicsMaterial"
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim or not mat_prim.IsValid():
        mat_prim = stage.DefinePrim(mat_path, "PhysicsMaterial")
    
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)
    
    # Bind to collision geometry
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        # Try to find collision prims and bind material
        for child in prim.GetAllChildren():
            if "collision" in child.GetName().lower() or child.HasAPI(UsdPhysics.CollisionAPI):
                child.CreateRelationship("physics:material:binding", custom=False).SetTargets([mat_prim.GetPath()])
        # Also bind to parent prim for broad coverage
        prim.CreateRelationship("physics:material:binding", custom=False).SetTargets([mat_prim.GetPath()])
    
    print(f"[FRICTION] Applied material to {prim_path}: static={static_friction}, dynamic={dynamic_friction}")

# Import utility functions from tasks module
import sys
TASKS_PATH = Path(__file__).parent / "../../../tasks"
sys.path.insert(0, str(TASKS_PATH.parent))
from tasks.utils import (
    parse_initial_joint_positions,
    parse_urdf_initial_joint_positions,
    parse_urdf_joint_dynamics,
    compute_object_spawn_position,
    parse_needs_disable_self_collision,
    parse_material_properties,
)

DATA_PATH = Path(__file__).parent / "../../assets"
OBJECTS_PATH = Path(__file__).parent / "../../objects"


def compute_obj_min_z(mesh_path: Path) -> float | None:
    """Compute the minimum z value from an OBJ mesh (vertex lines).

    Returns None if the file cannot be parsed.
    """
    try:
        min_z: float | None = None
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # OBJ vertex line: v x y z
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                z = float(parts[3])
                min_z = z if min_z is None else min(min_z, z)
        return min_z
    except Exception as e:
        print(f"[SCENE] Failed to compute OBJ min_z for {mesh_path}: {e}")
        return None


def compute_mesh_spawn_z(mesh_path: Path, table_offset: float = 0.005, scale: float = 1.0) -> float:
    """Compute a spawn z so the bottom of the mesh sits `table_offset` above z=0."""
    min_z = compute_obj_min_z(mesh_path)
    if min_z is None:
        return table_offset
    return -(min_z * scale) + table_offset


def _apply_red_material_to_cube(stage, cube_prim_path: str):
    """Apply red material to the cube, similar to franka_droid_sim2real_env.py"""
    if stage is None:
        return
    
    # Red color: (1.0, 0.0, 0.0) - solid red
    color = (1.0, 0.0, 0.0)
    metallic = 0.0
    roughness = 0.9  # matte
    
    # Create material path
    material_path = "/World/Looks/DroidCubeRedMaterial"
    
    # Get or create material
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
    material = UsdShade.Material(material_prim)
    
    # Create UsdPreviewSurface shader
    shader_path = f"{material_path}/PreviewSurface"
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim or not shader_prim.IsValid():
        shader_prim = stage.DefinePrim(shader_path, "Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    # Set material properties
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    
    # Bind material to cube with strong binding
    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    if cube_prim and cube_prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )


def _resolve_usd_path(path: str, *, assets_dir: Path) -> Path:
    """Resolve a USD path that can be absolute or relative to the sim-evals assets dir."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (assets_dir / p).resolve()


def _make_scene_without_objects(
    base_scene_usd: Path,
    *,
    prims_to_remove: list[str],
    assets_dir: Path,
) -> Path:
    """Create a patched USD scene that removes specified prims (cube, bowl, block, etc.).

    The patched file is cached under `assets/_generated_scenes/`.
    Uses flattening to preserve all sublayers and references (table, background, lighting).
    """
    base_scene_usd = base_scene_usd.resolve()
    out_dir = (assets_dir / "_generated_scenes").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    key = f"{base_scene_usd}|removed_{'_'.join(prims_to_remove)}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    out_path = out_dir / f"{base_scene_usd.stem}__no_objects__{digest}.usd"
    if out_path.exists():
        return out_path

    stage = Usd.Stage.Open(str(base_scene_usd))
    if stage is None:
        raise FileNotFoundError(f"Failed to open base scene USD: {base_scene_usd}")

    # Remove specified prims
    removed_any = False
    for prim_name in prims_to_remove:
        prim_path = f"/World/{prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            print(f"[SCENE] Removing prim: {prim_path}")
            stage.RemovePrim(prim_path)
            removed_any = True

    if not removed_any:
        # If nothing to remove, just use the original scene
        print(f"[SCENE] No prims to remove, using original: {base_scene_usd}")
        return base_scene_usd

    # Flatten and export to preserve all sublayers (table, background, lighting)
    flattened_layer = stage.Flatten()
    flattened_layer.Export(str(out_path))
    print(f"[SCENE] Created patched scene (flattened): {out_path}")
    return out_path


def _make_patched_scene_usd(
    base_scene_usd: Path,
    *,
    replace_prim_name: str,
    replacement_usd: Path,
    assets_dir: Path,
) -> Path:
    """Create a patched USD scene that replaces `/World/<replace_prim_name>` with a new USD reference.

    This is intentionally simple and keeps the original scene layout intact, only swapping the referenced
    asset at the specified prim path. The patched file is cached under `assets/_generated_scenes/`.
    """
    base_scene_usd = base_scene_usd.resolve()
    replacement_usd = replacement_usd.resolve()
    out_dir = (assets_dir / "_generated_scenes").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    key = f"{base_scene_usd}|{replace_prim_name}|{replacement_usd}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    out_path = out_dir / f"{base_scene_usd.stem}__replace_{replace_prim_name}__{digest}.usd"
    if out_path.exists():
        return out_path

    stage = Usd.Stage.Open(str(base_scene_usd))
    if stage is None:
        raise FileNotFoundError(f"Failed to open base scene USD: {base_scene_usd}")

    prim_path = f"/World/{replace_prim_name}"
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise ValueError(
            f"Prim '{prim_path}' not found in scene '{base_scene_usd.name}'. "
            f"Try a different --object_prim (look under /World in the USD)."
        )

    # Preserve the original transform if present.
    xformable = UsdGeom.Xformable(prim)
    translate = prim.GetAttribute("xformOp:translate").Get() if prim.HasAttribute("xformOp:translate") else None
    orient = prim.GetAttribute("xformOp:orient").Get() if prim.HasAttribute("xformOp:orient") else None

    # Replace the prim with a new Xform prim that references the replacement asset.
    stage.RemovePrim(prim_path)
    new_prim = stage.DefinePrim(prim_path, "Xform")
    new_prim.GetReferences().AddReference(str(replacement_usd))

    if translate is not None:
        new_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(translate)
    if orient is not None:
        # Preserve quaternion type if possible; fall back to Quatf.
        new_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf).Set(orient)

    stage.GetRootLayer().Export(str(out_path))
    return out_path


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=1080,
        width=1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

    # Recording camera: positioned on the left of the robot base, facing the table
    # Resolution: 1440p (2560x1440)
    recording_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/recording_cam",
        height=1440,
        width=2560,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.3, 0.85, 0.75), rot=(-0.35, -0.25, 0.45, 0.78), convention="opengl"
        ),
    )

    # -------------------------------------------------------------------------
    # Scene/object overrides
    # -------------------------------------------------------------------------
    # If set, overrides which scene USD is loaded (absolute path or relative to assets/).
    scene_usd_path: str | None = None
    # If set, replaces the rigid-body prim `/World/<replace_object_prim>` inside the chosen scene USD with this USD.
    replace_object_prim: str | None = None
    replace_object_usd_path: str | None = None

    def dynamic_scene(self, scene_name: str):
        # Resolve base scene file.
        if self.scene_usd_path is not None:
            environment_path = _resolve_usd_path(self.scene_usd_path, assets_dir=DATA_PATH)
        else:
            environment_path = DATA_PATH / f"scene{scene_name}.usd"

        # Optionally patch the scene to swap a specific object prim.
        if self.replace_object_usd_path is not None:
            if self.replace_object_prim is None:
                raise ValueError("replace_object_usd_path was set but replace_object_prim is None.")
            environment_path = _make_patched_scene_usd(
                environment_path,
                replace_prim_name=self.replace_object_prim,
                replacement_usd=_resolve_usd_path(self.replace_object_usd_path, assets_dir=DATA_PATH),
                assets_dir=DATA_PATH,
            )

        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=str(environment_path),
                    ),
                )
        self.scene = scene

        # We only inspect prim metadata (names/transforms) to register rigid bodies below.
        # Avoid loading payloads/references here to prevent noisy warnings (e.g. missing robot payloads
        # inside scene USDs) and to keep this lightweight.
        stage = Usd.Stage.Open(str(environment_path), load=Usd.Stage.LoadNone)
        scene_prim = stage.GetPrimAtPath("/World")
        children = scene_prim.GetChildren()

        found_cube = False
        for child in children:
            # if rigid body
            if not UsdPhysics.RigidBodyAPI(child):
                continue

            name = child.GetName()
            # Only register cube/block/object, skip other objects (bowls, mugs, etc.)
            if name.lower() not in ["cube", "block", "object"]:
                print(f"Skipping rigid body: {name} (not a cube/block/object)")
                continue
            
            print(f"Found rigid body: {name}")
            found_cube = True
            pos = child.GetAttribute("xformOp:translate").Get()
            rot = child.GetAttribute("xformOp:orient").Get()
            rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
            asset = RigidObjectCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                        spawn=None,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=pos,
                            rot=rot,
                        ),
                    )
            setattr(self, name, asset)
        
        # If no cube/block found in scene, add a default red cube
        if not found_cube:
            print("No cube found in scene, adding default red cube")
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/cube",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5, 0.0, 0.055),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            self.cube = cube

    def dynamic_scene_with_object(
        self,
        scene_name: str,
        object_name: str,
        object_pos: tuple | None = None,
        object_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        fix_base: bool = False,
        keep_bowl: bool = False,
        bowl_pos: tuple | None = None,
    ):
        """Load scene USD (with table/background) but replace cube/block/bowl with a URDF object.
        
        Args:
            scene_name: Scene number (e.g., "1" for scene1.usd)
            object_name: Object folder name in sim-evals/objects/ (e.g., '103275_evogen')
            object_pos: Initial position (x, y, z). If None, computed from bounding_box.json
            object_rot: Initial rotation quaternion (w, x, y, z)
            fix_base: If True, fix the object's base in place (no physics on root)
            keep_bowl: If True, add a bowl to the scene
            bowl_pos: Bowl position (x, y, z). If None, uses default (0.5, 0.15, 0.0)
        """
        # Load the scene USD (table, background, lighting are included)
        if self.scene_usd_path is not None:
            environment_path = _resolve_usd_path(self.scene_usd_path, assets_dir=DATA_PATH)
        else:
            environment_path = DATA_PATH / f"scene{scene_name}.usd"

        # Remove cube/block/bowl from the scene USD.
        # Note: The original bowl prim in `scene1.usd` is `_24_bowl` and references a remote
        # Isaac asset URL that is often unavailable / returns 404. We therefore remove it and
        # (optionally) spawn a local YCB 024 bowl (downloaded from ycb-benchmarks) at `bowl_pos`.
        prims_to_remove = [
            "rubiks_cube",  # actual cube name in scene1.usd
            "cube", "block", "can", "banana", "mug", "bin",  # generic names
            "Cube", "Block",  # capitalized variants
            "_24_bowl", "bowl", "Bowl",  # Always remove - we'll spawn our own
        ]
        
        print(f"[SCENE] keep_bowl={keep_bowl}, bowl_pos={bowl_pos}, prims_to_remove={prims_to_remove}")
        
        environment_path = _make_scene_without_objects(
            environment_path,
            prims_to_remove=prims_to_remove,
            assets_dir=DATA_PATH,
        )

        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(environment_path)),
        )
        self.scene = scene

        # Find the URDF file for the object
        object_folder = OBJECTS_PATH / object_name
        if not object_folder.exists():
            available = [d.name for d in OBJECTS_PATH.iterdir() if d.is_dir()]
            raise FileNotFoundError(f"Object folder not found: {object_folder}\nAvailable: {available}")
        
        # Try URDF files in order of preference
        urdf_candidates = [
            "mobility_revised.urdf",  # Refined URDF (evogen, ivw, vlm1shot)
            "mobility_human.urdf",    # Human-annotated (humangensim2)
            "mobility.urdf",          # Original/fallback
        ]
        urdf_path = None
        for urdf_name in urdf_candidates:
            candidate = object_folder / urdf_name
            if candidate.exists():
                urdf_path = candidate
                break
        if urdf_path is None:
            raise FileNotFoundError(f"No URDF found in {object_folder}. Tried: {urdf_candidates}")
        
        # Detect object type based on folder name suffix
        is_humangensim2 = object_name.endswith("_humangensim2")
        object_type = "humangensim2" if is_humangensim2 else "refined"
        
        # For humangensim2 objects, read scale from info.json (URDF meshes don't have embedded scale)
        # For refined objects (evogen/ivw/vlm1shot), scale is embedded in URDF mesh tags, so use 1.0
        object_scale = 1.0
        scale_source = "embedded in URDF"
        if is_humangensim2:
            info_path = object_folder / "info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                object_scale = info.get("scale", 1.0)
                scale_source = "info.json"
                print(f"[DROID-Eval] humangensim2 object: loaded scale={object_scale} from info.json")
        
        # Compute spawn position from bounding box (like franka_grasping_env.py)
        # Handle case where object_pos has z=None (compute z from bounding box, use provided x, y)
        # IMPORTANT: For humangensim2 objects, pass the scale from info.json so position is calculated
        # AFTER scaling (bounding_box.json contains UNSCALED bounds, and spawn position must account
        # for the scale that will be applied to the URDF at spawn time).
        scale_for_position = object_scale if is_humangensim2 else None
        # Table surface is at z≈0.045 in scene USD, so use table_offset=0.05 to place objects on top
        if object_pos is None:
            object_pos = compute_object_spawn_position(object_folder, table_offset=0.05, env_name="DROID-Eval", scale_override=scale_for_position)
        elif object_pos[2] is None:
            # Use provided x, y but compute z from bounding box
            computed_pos = compute_object_spawn_position(object_folder, table_offset=0.05, env_name="DROID-Eval", scale_override=scale_for_position)
            object_pos = (object_pos[0], object_pos[1], computed_pos[2])
            print(f"[DROID-Eval] Using x,y from config, z from bounding box: {object_pos}")
        
        # Parse initial joint positions
        # For humangensim2: read from URDF comments (<!-- INITIAL JOINT POSITIONS ... -->)
        # For refined (evogen/ivw/vlm1shot): read from overlay_refined.json
        if is_humangensim2:
            initial_joint_pos = parse_urdf_initial_joint_positions(str(urdf_path))
            if initial_joint_pos:
                print(f"[DROID-Eval] humangensim2: loaded initial joint positions from URDF comments: {initial_joint_pos}")
        else:
            initial_joint_pos = parse_initial_joint_positions(object_folder, env_name="DROID-Eval")
        
        # Store initial joint positions in module-level variable for the reset event
        # This is needed because InitialStateCfg.joint_pos doesn't work reliably for URDF articulations
        global _OBJECT_INITIAL_JOINT_POS
        _OBJECT_INITIAL_JOINT_POS = initial_joint_pos if initial_joint_pos else {}
        
        # Parse URDF dynamics (stiffness, damping, friction) for verification
        urdf_dynamics = parse_urdf_joint_dynamics(str(urdf_path))
        
        # Check if self-collision needs to be disabled
        disable_self_collision = parse_needs_disable_self_collision(object_folder, env_name="DROID-Eval")
        
        # Parse material properties from overlay (friction, restitution)
        # Store in module-level variable (not on self, which IsaacLab parses as asset config)
        global _OBJECT_MATERIAL_PROPS
        _OBJECT_MATERIAL_PROPS = parse_material_properties(object_folder, env_name="DROID-Eval")
        
        print(f"\n{'='*60}")
        print(f"[DROID-Eval] OBJECT CONFIGURATION")
        print(f"{'='*60}")
        print(f"Object: {object_name}")
        print(f"Object Type: {object_type}")
        print(f"URDF: {urdf_path}")
        print(f"Scene: {environment_path}")
        print(f"Scale: {object_scale} (source: {scale_source})")
        print(f"\nSpawn Position: {object_pos}")
        print(f"Spawn Rotation: {object_rot}")
        print(f"\nInitial Joint Positions (from URDF comment):")
        if initial_joint_pos:
            for jname, jpos in initial_joint_pos.items():
                print(f"  {jname}: {jpos:.4f} rad ({np.degrees(jpos):.2f}°)")
        else:
            print(f"  (none specified)")
        print(f"\nURDF Joint Dynamics (parsed from URDF):")
        print(f"  Stiffness: {urdf_dynamics['stiffness'] or '(not specified - using simulator default)'}")
        print(f"  Damping: {urdf_dynamics['damping'] or '(not specified - using simulator default)'}")
        print(f"  Friction: {urdf_dynamics['friction'] or '(not specified - using simulator default)'}")
        print(f"\nSelf-collision enabled: {not disable_self_collision}")
        print(f"Fix base: {fix_base}")
        print(f"\nMaterial Properties (from overlay):")
        print(f"  Friction: {_OBJECT_MATERIAL_PROPS['friction']}")
        print(f"  Restitution: {_OBJECT_MATERIAL_PROPS['restitution']}")
        print(f"{'='*60}\n")
        
        # Build actuator config with URDF dynamics values
        # ImplicitActuatorCfg accepts per-joint dicts for stiffness, damping, friction
        # 
        # Use None when not specified to let simulator use its defaults.
        # The reset_object_joints event will set the correct initial position.
        actuator_stiffness = urdf_dynamics['stiffness'] if urdf_dynamics['stiffness'] else None
        actuator_damping = urdf_dynamics['damping'] if urdf_dynamics['damping'] else None
        actuator_friction = urdf_dynamics['friction'] if urdf_dynamics['friction'] else None
        
        # Add URDF object with explicit actuator dynamics from URDF
        self.object = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=list(object_pos),
                rot=list(object_rot),
                joint_pos=initial_joint_pos if initial_joint_pos else {},
            ),
            spawn=UrdfFileCfg(
                asset_path=str(urdf_path),
                root_link_name="base",
                force_usd_conversion=True,
                collider_type="convex_decomposition",  # Enable convex decomposition for accurate collision geometry
                scale=(object_scale, object_scale, object_scale),  # For humangensim2: from info.json. For refined: 1.0 (scale embedded in URDF)
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=not disable_self_collision,
                ),
                activate_contact_sensors=False,
                fix_base=fix_base,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=actuator_stiffness,
                        damping=actuator_damping,
                    )
                ),
            ),
            actuators={
                # Use URDF dynamics for joint behavior (passive spring-damper)
                "object_actuator": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=actuator_stiffness,
                    damping=actuator_damping,
                    friction=actuator_friction,
                ),
            },
        )
        
        # Spawn the YCB 024 bowl (local meshes) with configurable position.
        if keep_bowl:
            # Default bowl position: to the right of center (y > 0). z=None => auto-compute to sit on table.
            if bowl_pos is None:
                bowl_pos = (0.5, 0.15, None)

            # Use a smooth scanned bowl mesh (Libero stable_scanned_objects) which looks closer to a
            # real plastic bowl than the faceted YCB reconstruction.
            bowl_assets_dir = (DATA_PATH / "red_bowl").resolve()
            bowl_urdf_path = bowl_assets_dir / "bowl.urdf"
            bowl_mesh_path = bowl_assets_dir / "model.obj"

            # Bowl mesh is scaled at spawn-time (more reliable than relying on URDF mesh scale tags).
            bowl_mesh_scale = 0.7

            if bowl_pos[2] is None:
                # Table surface is around z=0.04-0.05. Place bowl just on top.
                bowl_z = compute_mesh_spawn_z(bowl_mesh_path, table_offset=0.048, scale=bowl_mesh_scale)
                print(f"[SCENE] Bowl z auto-computed from mesh bounds: z={bowl_z:.4f}")
            else:
                bowl_z = float(bowl_pos[2])

            print(f"[SCENE] Spawning bowl at position: ({bowl_pos[0]}, {bowl_pos[1]}, {bowl_z})")
            print(f"[SCENE] Bowl URDF: {bowl_urdf_path}")

            self.bowl = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/bowl",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(bowl_pos[0], bowl_pos[1], bowl_z),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    # Bowl URDF has no joints. Override defaults (which use regex ".*") to avoid
                    # failing name-resolution on an empty joint list.
                    joint_pos={},
                    joint_vel={},
                ),
                spawn=UrdfFileCfg(
                    asset_path=str(bowl_urdf_path),
                    root_link_name="base",
                    force_usd_conversion=True,
                    # Fixed-base bowl: allow concave mesh collision so objects can land inside the bowl.
                    collider_type="trimesh",
                    scale=(bowl_mesh_scale, bowl_mesh_scale, bowl_mesh_scale),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=True,
                    ),
                    fix_base=True,
                    joint_drive=UrdfConverterCfg.JointDriveCfg(
                        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                            stiffness=0.0,
                            damping=0.0,
                        )
                    ),
                ),
                actuators={},
            )


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction

@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        recording_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("recording_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Custom reset event to write object joint positions from URDF comments
    # This is needed because InitialStateCfg.joint_pos doesn't work reliably for URDF articulations
    reset_object_joints = EventTerm(
        func=reset_object_joints_to_initial,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object")},
    )

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    
    # Global physics material with higher friction for gripper-object interaction
    # Robotiq 2F-85 has rubber finger pads, so we use friction ~1.0 (not 0.5 default)
    # Using "average" combine mode: effective friction = (gripper + object) / 2
    # Example: rubber gripper (1.0) + ABS plastic stapler (0.5) → 0.75 effective
    sim: SimulationCfg = SimulationCfg(
        dt=1 / (60 * 2),
        render_interval=4 * 2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",  # Use max friction for better grip
            restitution_combine_mode="multiply",
            static_friction=DEFAULT_STATIC_FRICTION,
            dynamic_friction=DEFAULT_DYNAMIC_FRICTION,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        ),
    )

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        # Note: sim.dt and sim.render_interval are now set in SimulationCfg above

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True
        
        print(f"[ENV] Physics material: static_friction={DEFAULT_STATIC_FRICTION}, dynamic_friction={DEFAULT_DYNAMIC_FRICTION}")

    
    def set_scene(self, scene_name: str):
        self.scene.dynamic_scene(scene_name)

    def set_scene_with_object(
        self,
        object_name: str,
        scene_name: str = "1",
        object_pos: tuple | None = None,
        object_rot: tuple = (1.0, 0.0, 0.0, 0.0),
        fix_base: bool = False,
        keep_bowl: bool = False,
        bowl_pos: tuple | None = None,
    ):
        """Configure scene with URDF object instead of cube/block/bowl.
        
        Args:
            object_name: Object folder name in sim-evals/objects/ (e.g., '103275_evogen')
            scene_name: Scene number (default "1" for scene1.usd with table/background)
            object_pos: Initial position (x, y, z). If None, computed from bounding_box.json
            object_rot: Initial rotation quaternion (w, x, y, z)
            fix_base: If True, fix the object's base in place (no physics on root)
            keep_bowl: If True, add a bowl to the scene
            bowl_pos: Bowl position (x, y, z). If None, uses default (0.5, 0.15, 0.0)
        """
        self.scene.dynamic_scene_with_object(
            scene_name=scene_name,
            object_name=object_name,
            object_pos=object_pos,
            object_rot=object_rot,
            fix_base=fix_base,
            keep_bowl=keep_bowl,
            bowl_pos=bowl_pos,
        )


