"""
Depth score computation and proximity-based tier resolution.

Depth Anything V2 outputs relative disparity — not metric distance.
We use percentile-based normalization to build a proximity classifier
that works without calibration hardware.
"""

import numpy as np

from class_tiers import CLASS_TO_BASE_TIER, TIER_ORDER
from config import DEPTH_VERY_CLOSE, DEPTH_CLOSE, DEPTH_MEDIUM


def get_depth_score(depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Compute a normalized depth score for a detected object.

    Args:
        depth_map: 2D numpy array of depth values (higher = closer in disparity).
        bbox: (x1, y1, x2, y2) in depth_map coordinates.

    Returns:
        Normalized score 0.0–1.0. Lower = closer to camera.
    """
    x1, y1, x2, y2 = bbox

    # Clamp to depth map bounds
    h, w = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 1.0  # degenerate bbox → treat as far away

    region = depth_map[y1:y2, x1:x2]

    if region.size == 0:
        return 1.0

    # Depth Anything outputs disparity: higher value = closer to camera.
    # 90th percentile catches the closest surface of the object inside the bounding box.
    object_depth = np.percentile(region, 90)

    frame_min = depth_map.min()
    frame_max = depth_map.max()
    
    # Invert so 0.0 is closest and 1.0 is furthest
    normalized = (frame_max - object_depth) / (frame_max - frame_min + 1e-6)

    return float(normalized)  # lower = closer


def resolve_tier(depth_score: float, object_class: str) -> str | None:
    """
    Escalate or suppress the base tier based on proximity.

    Args:
        depth_score: 0.0–1.0, lower = closer.
        object_class: YOLO class name.

    Returns:
        Resolved tier string (P0–P4) or None if the alert should be suppressed.
    """
    base_tier = CLASS_TO_BASE_TIER.get(object_class)
    if base_tier is None:
        return None  # unknown class → no alert

    if depth_score < DEPTH_VERY_CLOSE:
        # Very close — override to P0 regardless of class
        return "P0"
    elif depth_score < DEPTH_CLOSE:
        # Close — escalate one tier
        idx = TIER_ORDER.index(base_tier)
        return TIER_ORDER[max(0, idx - 1)]
    elif depth_score < DEPTH_MEDIUM:
        # Medium distance — use base tier as-is
        return base_tier
    else:
        # Far — suppress alert
        return None


def find_generic_obstacles(depth_map: np.ndarray) -> list[dict]:
    """
    Split the depth map into 3 vertical zones (LEFT, CENTER, RIGHT).
    If any zone has a large cluster of pixels that is very close,
    return it as a generic obstacle.
    """
    h, w = depth_map.shape[:2]
    third = w // 3
    
    # Crop the bottom 30% of the image to ignore the floor right in front of the user
    crop_h = int(h * 0.7)
    cropped_map = depth_map[:crop_h, :]
    
    zones = {
        "LEFT": cropped_map[:, :third],
        "CENTER": cropped_map[:, third:2*third],
        "RIGHT": cropped_map[:, 2*third:]
    }

    # Use the full depth map min/max for normalization so it matches YOLO objects
    frame_min = depth_map.min()
    frame_max = depth_map.max()

    obstacles = []
    
    for zone_name, region in zones.items():
        if region.size == 0:
            continue
            
        # 95th percentile catches the absolute closest surface in that region
        zone_closest = np.percentile(region, 95)
        # 50th percentile represents the background/average depth of the zone
        zone_median = np.percentile(region, 50)
        
        # Normalize relative to the whole frame
        normalized_closest = (frame_max - zone_closest) / (frame_max - frame_min + 1e-6)
        
        # How much does this object "stick out" from its background?
        # Difference between closest point and median point
        prominence = (zone_closest - zone_median) / (frame_max - frame_min + 1e-6)
        
        # Require stricter thresholds for generic obstacles to avoid spam
        threshold = 0.25 if zone_name == "CENTER" else 0.15
        
        # ONLY alert if it's close AND it significantly sticks out from the background
        # (prominence > 0.15 means it's a distinct object, not just a slightly slanted wall)
        if normalized_closest < threshold and prominence > 0.15:
            # Only objects directly in the CENTER can trigger a P0 (Immediate Stop). 
            # Side objects are capped at P1 (Slow down / Warning).
            tier = "P0" if (normalized_closest < DEPTH_VERY_CLOSE and zone_name == "CENTER") else "P1"
            
            obstacles.append({
                "tier": tier,
                "label": "obstacle",
                "zone": zone_name,
                "depth_score": float(normalized_closest)
            })
            
    return obstacles

