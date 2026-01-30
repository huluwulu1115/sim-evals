"""
Configuration for suitcase object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Suitcase Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Close the suitcase lid."

# Initial spawn rotation (quaternion: w, x, y, z)
# Slight rotation around Y-axis (-15 degrees) for stability
INITIAL_ROTATION = (0.991, 0.0, -0.131, 0.0)

# Tool use: joint_0 is the lid hinge
# Joint limits: lower=-1.57 rad, upper=1.57 rad
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = -1.047  # Target position when closed (radians)
TOOL_USE_TOLERANCE = 0.2  # Position tolerance (radians)


# Scene layout - move suitcase further from arm in positive x direction
TOOL_USE_OBJECT_POS = (0.55, 0.0, None)  # z=None means auto-compute from bounding box

# Shared task configs (suitcase only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
    fix_base=True,  # Suitcase can move freely
    object_pos=TOOL_USE_OBJECT_POS,
)


# =============================================================================
# Suitcase Object Configurations
# =============================================================================

SUITCASE_CONFIGS = {
    "103755_evogen": ObjectTaskConfig(
        object_name="103755_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103755_humangensim2": ObjectTaskConfig(
        object_name="103755_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103755_ivw": ObjectTaskConfig(
        object_name="103755_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "103755_vlm1shot": ObjectTaskConfig(
        object_name="103755_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific suitcase variant.
    
    Args:
        variant: One of '103755_evogen', '103755_humangensim2', '103755_ivw', '103755_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in SUITCASE_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(SUITCASE_CONFIGS.keys())}")
    return SUITCASE_CONFIGS[variant]
