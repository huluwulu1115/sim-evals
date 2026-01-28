"""
Configuration for toilet object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Toilet Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Close the toilet lid."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Tool use: joint_0 is the lid hinge
# Joint limits: lower=0.0 rad (closed), upper=0.52 rad (~30°, open)
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = 0.05  # Target position when closed (radians)
TOOL_USE_TOLERANCE = 0.1  # Position tolerance (radians)


# Shared task configs (toilet only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Toilet Object Configurations
# =============================================================================

TOILET_CONFIGS = {
    "102703_evogen": ObjectTaskConfig(
        object_name="102703_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102703_humangensim2": ObjectTaskConfig(
        object_name="102703_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102703_ivw": ObjectTaskConfig(
        object_name="102703_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "102703_vlm1shot": ObjectTaskConfig(
        object_name="102703_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific toilet variant.
    
    Args:
        variant: One of '102703_evogen', '102703_humangensim2', '102703_ivw', '102703_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in TOILET_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(TOILET_CONFIGS.keys())}")
    return TOILET_CONFIGS[variant]
