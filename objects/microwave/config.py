"""
Configuration for microwave object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Microwave Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Close the microwave door."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Tool use: joint_0 is the door hinge
# Joint limits: lower=0.0 rad (closed), upper=1.57 rad (90°, open)
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = 0.1  # Target position when closed (radians)
TOOL_USE_TOLERANCE = 0.2  # Position tolerance (radians)


# Shared task configs (microwave only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Microwave Object Configurations
# =============================================================================

MICROWAVE_CONFIGS = {
    "7236_evogen": ObjectTaskConfig(
        object_name="7236_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "7236_humangensim2": ObjectTaskConfig(
        object_name="7236_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "7236_ivw": ObjectTaskConfig(
        object_name="7236_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "7236_vlm1shot": ObjectTaskConfig(
        object_name="7236_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific microwave variant.
    
    Args:
        variant: One of '7236_evogen', '7236_humangensim2', '7236_ivw', '7236_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in MICROWAVE_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(MICROWAVE_CONFIGS.keys())}")
    return MICROWAVE_CONFIGS[variant]
