"""
Configuration for trashcan object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Trashcan Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Close the trashcan lid."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Tool use: joint_0 is the lid hinge
# Joint limits: lower=0.0 rad (closed), upper=0.57 rad (~33°, open)
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = 0.05  # Target position when closed (radians)
TOOL_USE_TOLERANCE = 0.1  # Position tolerance (radians)


# Shared task configs (trashcan only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Trashcan Object Configurations
# =============================================================================

TRASHCAN_CONFIGS = {
    "102181_evogen": ObjectTaskConfig(
        object_name="102181_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102181_humangensim2": ObjectTaskConfig(
        object_name="102181_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102181_ivw": ObjectTaskConfig(
        object_name="102181_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102181_vlm1shot": ObjectTaskConfig(
        object_name="102181_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific trashcan variant.
    
    Args:
        variant: One of '102181_evogen', '102181_humangensim2', '102181_ivw', '102181_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in TRASHCAN_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(TRASHCAN_CONFIGS.keys())}")
    return TRASHCAN_CONFIGS[variant]
