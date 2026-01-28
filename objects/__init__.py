"""
Universal configuration and success criteria for object tasks.

Each object can have two task types:
- grasping: Pick up the object (success = raised 8cm or more)
- tool_use: Manipulate the object's articulation (success = joint reaches target position)
"""

from dataclasses import dataclass


@dataclass
class GraspingTaskConfig:
    """Configuration for grasping task."""
    prompt: str
    min_lift_height: float = 0.08  # 8cm default success threshold
    # Scene layout: object position (x, y, z) - if None, computed from bounding box
    object_pos: tuple[float, float, float] | None = None
    # Whether to include the bowl in the scene (for pick-and-place tasks)
    keep_bowl: bool = False
    # Bowl position (x, y, z) - if None, uses default. Only used if keep_bowl=True
    bowl_pos: tuple[float, float, float] | None = None


@dataclass
class ToolUseTaskConfig:
    """Configuration for tool use task."""
    prompt: str
    target_joint: str  # Name of the joint to manipulate
    target_position: float  # Target position in radians
    position_tolerance: float = 0.1  # Tolerance for success (radians)


@dataclass
class ObjectTaskConfig:
    """Configuration for all tasks on a specific object variant."""
    object_name: str
    initial_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # Quaternion (w, x, y, z)
    grasping: GraspingTaskConfig | None = None
    tool_use: ToolUseTaskConfig | None = None


def check_grasping_success(
    initial_z: float,
    current_z: float,
    config: GraspingTaskConfig,
) -> bool:
    """Check if grasping task succeeded (object lifted by threshold).
    
    Args:
        initial_z: Initial z-position of object
        current_z: Current z-position of object
        config: Grasping task configuration
    
    Returns:
        True if object was lifted by at least min_lift_height
    """
    lift_height = current_z - initial_z
    return lift_height >= config.min_lift_height


def check_tool_use_success(
    joint_positions: dict[str, float],
    config: ToolUseTaskConfig,
) -> bool:
    """Check if tool use task succeeded (joint reached target position).
    
    Args:
        joint_positions: Dict mapping joint names to current positions (radians)
        config: Tool use task configuration
    
    Returns:
        True if target joint is within tolerance of target position
    """
    if config.target_joint not in joint_positions:
        return False
    
    current_pos = joint_positions[config.target_joint]
    error = abs(current_pos - config.target_position)
    return error <= config.position_tolerance
