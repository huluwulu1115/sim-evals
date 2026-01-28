"""
Configuration for laptop object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Laptop Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Fold the laptop closed."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Tool use: joint_1 is the screen hinge
# Joint limits: lower=-1.85 rad (closed), upper=0.0 rad (open)
TOOL_USE_TARGET_JOINT = "joint_1"
TOOL_USE_TARGET_POSITION = -1.5  # Target position when folded (radians)
TOOL_USE_TOLERANCE = 0.2  # Position tolerance (radians)


# Shared task configs (laptop only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Laptop Object Configurations
# =============================================================================

LAPTOP_CONFIGS = {
    "9912_evogen": ObjectTaskConfig(
        object_name="9912_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "9912_humangensim2": ObjectTaskConfig(
        object_name="9912_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "9912_ivw": ObjectTaskConfig(
        object_name="9912_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "9912_vlm1shot": ObjectTaskConfig(
        object_name="9912_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific laptop variant.
    
    Args:
        variant: One of '9912_evogen', '9912_humangensim2', '9912_ivw', '9912_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in LAPTOP_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(LAPTOP_CONFIGS.keys())}")
    return LAPTOP_CONFIGS[variant]
