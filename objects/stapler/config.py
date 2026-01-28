"""
Configuration for stapler object tasks.
"""

from objects import (
    ObjectTaskConfig,
    GraspingTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Stapler Task Configuration (shared across all variants)
# =============================================================================

# Prompts
GRASPING_PROMPT = "Pick up the stapler and put into the bowl."
# Prompts
TOOL_USE_PROMPT = "Press down on the stapler."

# Initial spawn rotation (quaternion: w, x, y, z)
# 30° rotation around positive Z axis
INITIAL_ROTATION = (0.9659, 0.0, 0.0, 0.2588)

# Success criteria
GRASPING_LIFT_HEIGHT = 0.08

# Grasping scene layout: stapler on left, bowl on right
# y < 0 = left, y > 0 = right (from robot's perspective)
GRASPING_OBJECT_POS = (0.5, -0.12, None)  # z=None means auto-compute from bounding box
GRASPING_BOWL_POS = (0.5, 0.18, None)  # z=None means auto-compute so bowl sits on table

# Tool use: joint_1 is the pressing joint
TOOL_USE_TARGET_JOINT = "joint_1"
TOOL_USE_TARGET_POSITION = -0.11  # Target position when pressed (radians)
TOOL_USE_TOLERANCE = 0.1  # Position tolerance (radians)


# Shared task configs
_GRASPING_CONFIG = GraspingTaskConfig(
    prompt=GRASPING_PROMPT,
    min_lift_height=GRASPING_LIFT_HEIGHT,
    object_pos=GRASPING_OBJECT_POS,
    keep_bowl=True,  # Include bowl for pick-and-place
    bowl_pos=GRASPING_BOWL_POS,
)

_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Stapler Object Configurations
# =============================================================================

STAPLER_CONFIGS = {
    "103275_evogen": ObjectTaskConfig(
        object_name="103275_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103275_humangensim2": ObjectTaskConfig(
        object_name="103275_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103275_ivw": ObjectTaskConfig(
        object_name="103275_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103275_vlm1shot": ObjectTaskConfig(
        object_name="103275_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific stapler variant.
    
    Args:
        variant: One of '103275_evogen', '103275_humangensim2', '103275_ivw', '103275_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in STAPLER_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(STAPLER_CONFIGS.keys())}")
    return STAPLER_CONFIGS[variant]
