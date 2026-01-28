"""
Configuration for box object tasks.
"""

from objects import (
    ObjectTaskConfig,
    GraspingTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Box Task Configuration (shared across all variants)
# =============================================================================

# Prompts
GRASPING_PROMPT = "Pick up the box and put into the bowl."
TOOL_USE_PROMPT = "Close the box lid."

# Tool use task parameters
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = -1.047  # Closed position (radians)
TOOL_USE_TOLERANCE = 0.05  # 0.05 radians tolerance

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Success criteria
GRASPING_LIFT_HEIGHT = 0.08  # 8cm lift for grasping success

# Grasping scene layout
GRASPING_OBJECT_POS = (0.5, -0.12, None)  # z=None means auto-compute from bounding box
GRASPING_BOWL_POS = (0.5, 0.18, None)  # z=None means auto-compute so bowl sits on table


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
# Box Object Configurations
# =============================================================================

BOX_CONFIGS = {
    "100221_evogen": ObjectTaskConfig(
        object_name="100221_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100221_humangensim2": ObjectTaskConfig(
        object_name="100221_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100221_ivw": ObjectTaskConfig(
        object_name="100221_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100221_vlm1shot": ObjectTaskConfig(
        object_name="100221_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific box variant.
    
    Args:
        variant: One of '100221_evogen', '100221_humangensim2', '100221_ivw', '100221_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in BOX_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(BOX_CONFIGS.keys())}")
    return BOX_CONFIGS[variant]
