"""
Configuration for switch object tasks.
"""

from objects import (
    ObjectTaskConfig,
    ToolUseTaskConfig,
)


# =============================================================================
# Switch Task Configuration (shared across all variants)
# =============================================================================

# Prompts
TOOL_USE_PROMPT = "Flip the switch."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Tool use: joint_0 is the toggle button
# Joint limits: lower=0.0 rad (off), upper=0.314 rad (~18°, on)
TOOL_USE_TARGET_JOINT = "joint_0"
TOOL_USE_TARGET_POSITION = 0.25  # Target position when flipped (radians)
TOOL_USE_TOLERANCE = 0.1  # Position tolerance (radians)


# Shared task configs (switch only has tool_use, no grasping)
_TOOL_USE_CONFIG = ToolUseTaskConfig(
    prompt=TOOL_USE_PROMPT,
    target_joint=TOOL_USE_TARGET_JOINT,
    target_position=TOOL_USE_TARGET_POSITION,
    position_tolerance=TOOL_USE_TOLERANCE,
)


# =============================================================================
# Switch Object Configurations
# =============================================================================

SWITCH_CONFIGS = {
    "100952_evogen": ObjectTaskConfig(
        object_name="100952_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100952_humangensim2": ObjectTaskConfig(
        object_name="100952_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100952_ivw": ObjectTaskConfig(
        object_name="100952_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
    "100952_vlm1shot": ObjectTaskConfig(
        object_name="100952_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=None,
        tool_use=_TOOL_USE_CONFIG,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific switch variant.
    
    Args:
        variant: One of '100952_evogen', '100952_humangensim2', '100952_ivw', '100952_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in SWITCH_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(SWITCH_CONFIGS.keys())}")
    return SWITCH_CONFIGS[variant]
