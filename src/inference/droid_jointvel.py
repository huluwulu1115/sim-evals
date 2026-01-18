import numpy as np
from openpi_client import websocket_client_policy, image_tools

from .abstract_client import InferenceClient


class Client(InferenceClient):
    """Inference client for DROID-style *joint velocity* policies.

    Many DROID policies (including pi05_droid-style configs) output:
      - 7D normalized joint velocity commands in [-1, 1]
      - 1D gripper command in [0, 1] (0=open, 1=close)

    This simulator expects *joint position* targets. We convert velocity -> position by integrating:
        q_target[t+1] = q_target[t] + (v_norm * vel_limits) * dt
    """

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        dt: float = 1.0 / 15.0,
        vel_limits: tuple[float, float, float, float, float, float, float] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        *,
        debug_vlm: bool = False,
        vlm_head_dim: int = 16,
        vlm_full: bool = False,
    ) -> None:
        self.open_loop_horizon = int(open_loop_horizon)
        self.dt = float(dt)
        self.vel_limits = np.asarray(vel_limits, dtype=np.float32)
        self.debug_vlm = bool(debug_vlm)
        self.vlm_head_dim = int(vlm_head_dim)
        self.vlm_full = bool(vlm_full)

        self.client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._q_target = None  # type: ignore[assignment]

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self._q_target = None

    def infer(self, obs: dict, instruction: str) -> dict:
        curr_obs = self._extract_observation(obs)
        vlm = None

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
            or self.pred_action_chunk is None
        ):
            self.actions_from_chunk_completed = 0

            # Initialize open-loop integration anchor from the *current* joint position.
            self._q_target = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()

            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            if self.debug_vlm:
                request_data["__openpi_debug__"] = {
                    "vlm": True,
                    "vlm_head_dim": self.vlm_head_dim,
                    "vlm_full": self.vlm_full,
                }

            resp = self.client.infer(request_data)
            self.pred_action_chunk = resp["actions"]
            vlm = resp.get("vlm")

        action_raw = np.asarray(self.pred_action_chunk[self.actions_from_chunk_completed], dtype=np.float32)
        self.actions_from_chunk_completed += 1

        # Velocity -> position integration (7D)
        v_norm = np.clip(action_raw[:7], -1.0, 1.0)
        v_cmd = v_norm * self.vel_limits  # rad/s
        if self._q_target is None:
            self._q_target = np.asarray(curr_obs["joint_position"], dtype=np.float32).copy()
        self._q_target = self._q_target + v_cmd * self.dt

        # Gripper: keep [0,1] and binarize for the sim's BinaryJointPositionZeroToOneAction.
        gripper = float(action_raw[7]) if action_raw.shape[0] >= 8 else 0.0
        gripper = 1.0 if gripper > 0.5 else 0.0

        action = np.concatenate([self._q_target.astype(np.float32), np.array([gripper], dtype=np.float32)], axis=0)

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        viz = np.concatenate([img1, img2], axis=1)

        return {"action": action, "viz": viz, "vlm": vlm}

    def _extract_observation(self, obs_dict, *, save_to_disk: bool = False):
        # Assign images (H,W,3)
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        # Optional debugging capture
        if save_to_disk:
            from PIL import Image

            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


