import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

import hashlib
from typing import List
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
from pxr import Gf, Sdf, UsdShade

from .nvidia_droid import NVIDIA_DROID

DATA_PATH = Path(__file__).parent / "../../assets"


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
        height=720,
        width=1280,
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
        height=720,
        width=1280,
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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True

    
    def set_scene(self, scene_name: str):
        self.scene.dynamic_scene(scene_name)


