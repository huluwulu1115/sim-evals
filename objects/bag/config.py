"""
Configuration for bag object tasks.
"""

from objects import (
    ObjectTaskConfig,
    GraspingTaskConfig,
)


# =============================================================================
# Bag Task Configuration (shared across all variants)
# =============================================================================

# Prompts
GRASPING_PROMPT = "Pick up the bag and put into the bowl."

# Initial spawn rotation (quaternion: w, x, y, z)
INITIAL_ROTATION = (1.0, 0.0, 0.0, 0.0)

# Success criteria
GRASPING_LIFT_HEIGHT = 0.08  # 8cm lift for grasping success

# Grasping scene layout
GRASPING_OBJECT_POS = (0.5, -0.12, None)  # z=None means auto-compute from bounding box
GRASPING_BOWL_POS = (0.5, 0.18, 0.01)  # Bowl on right, z=1cm above table


# Shared task configs (bag only has grasping, no tool_use)
_GRASPING_CONFIG = GraspingTaskConfig(
    prompt=GRASPING_PROMPT,
    min_lift_height=GRASPING_LIFT_HEIGHT,
    object_pos=GRASPING_OBJECT_POS,
    bowl_pos=GRASPING_BOWL_POS,
)


# =============================================================================
# Bag Object Configurations
# =============================================================================

BAG_CONFIGS = {
    "101673_evogen": ObjectTaskConfig(
        object_name="101673_evogen",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=None,
    ),
    "101673_humangensim2": ObjectTaskConfig(
        object_name="101673_humangensim2",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=None,
    ),
    "101673_ivw": ObjectTaskConfig(
        object_name="101673_ivw",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=None,
    ),
    "101673_vlm1shot": ObjectTaskConfig(
        object_name="101673_vlm1shot",
        initial_rotation=INITIAL_ROTATION,
        grasping=_GRASPING_CONFIG,
        tool_use=None,
    ),
}


def get_config(variant: str) -> ObjectTaskConfig:
    """Get configuration for a specific bag variant.
    
    Args:
        variant: One of '101673_evogen', '101673_humangensim2', '101673_ivw', '101673_vlm1shot'
    
    Returns:
        ObjectTaskConfig for the specified variant
    """
    if variant not in BAG_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(BAG_CONFIGS.keys())}")
    return BAG_CONFIGS[variant]
