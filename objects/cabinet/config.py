"""
Configuration for cabinet object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Cabinet Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Close the drawer of the cabinet."

# Initial spawn rotation (quaternion: w, x, y, z)
# -90 degrees around Z axis to orient cabinet correctly
INITIAL_ROTATION = (0.7071, 0.0, 0.0, -0.7071)

# Tool use: joint_0 is the drawer (prismatic)
# Joint limits: lower=0.0 m (closed), upper=1.0 m (open)
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = 0.0  # Target position when fully closed (meters)
TOOL_USE_TOLERANCE = 0.1  # Position tolerance (meters)


# Shared task configs (cabinet only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Cabinet Object Configurations
# =============================================================================

CABINET_CONFIGS = {
    "19179_evogen": ObjectTaskConfig(
        object_name="19179_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "19179_humangensim2": ObjectTaskConfig(
        object_name="19179_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "19179_ivw": ObjectTaskConfig(
        object_name="19179_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "19179_vlm1shot": ObjectTaskConfig(
        object_name="19179_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific cabinet variant.
    
    Args:
        variant: One of '19179_evogen', '19179_humangensim2', '19179_ivw', '19179_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in CABINET_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(CABINET_CONFIGS.keys())}")
    return CABINET_CONFIGS[variant]
