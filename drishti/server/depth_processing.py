"""
Depth score computation and metric-based tier resolution.

Depth Anything V2 Metric outputs absolute depth in meters.
No normalization needed — thresholds are physical distances.
"""

import numpy as np

from class_tiers import CLASS_TO_BASE_TIER, TIER_ORDER
from config import METRIC_P0_M, METRIC_P1_M, METRIC_P2_M, METRIC_P3_M


def get_depth_meters(depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Returns 10th percentile depth in meters inside bbox.
    Lower = closer.

    Args:
        depth_map: 2D numpy array in meters.
        bbox: (x1, y1, x2, y2) in depth_map coordinates.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 999.0  # degenerate bbox → treat as very far

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 999.0

    return float(np.percentile(region, 10))


def resolve_tier_metric(depth_meters: float, object_class: str) -> str | None:
    """
    Determine alert tier from absolute distance in meters.

    Returns tier string (P0–P3) or None if too far to alert.
    """
    base_tier = CLASS_TO_BASE_TIER.get(object_class)
    if base_tier is None:
        return None

    # Determine tier from absolute distance
    if depth_meters < METRIC_P0_M:
        distance_tier = "P0"
    elif depth_meters < METRIC_P1_M:
        distance_tier = "P1"
    elif depth_meters < METRIC_P2_M:
        distance_tier = "P2"
    elif depth_meters < METRIC_P3_M:
        distance_tier = "P3"
    else:
        return None  # too far → suppress

    # Take the more dangerous of class-based and distance-based tier
    class_idx = TIER_ORDER.index(base_tier)
    distance_idx = TIER_ORDER.index(distance_tier)
    return TIER_ORDER[min(class_idx, distance_idx)]


def find_generic_obstacles(depth_map: np.ndarray) -> list[dict]:
    """
    Split depth map into 3 vertical zones (LEFT, CENTER, RIGHT).
    Alert if any zone has a significant close cluster.
    Now uses meters directly.
    """
    h, w = depth_map.shape[:2]
    third = w // 3

    # Crop top 30% (sky/ceiling)
    crop_h = int(h * 0.7)
    cropped = depth_map[:crop_h, :]

    zones = {
        "LEFT":   cropped[:, :third],
        "CENTER": cropped[:, third:2*third],
        "RIGHT":  cropped[:, 2*third:],
    }

    obstacles = []

    for zone_name, region in zones.items():
        if region.size == 0:
            continue

        # 10th percentile = closest surface in zone
        closest_m = float(np.percentile(region, 10))
        # 50th percentile = background depth
        median_m = float(np.percentile(region, 50))

        # Prominence: how much does the closest surface differ from background?
        prominence = median_m - closest_m

        # Thresholds in meters
        threshold = METRIC_P1_M if zone_name == "CENTER" else METRIC_P0_M

        # Alert if close AND prominent (distinct object, not a flat wall)
        if closest_m < threshold and prominence > 0.5:
            tier = "P0" if (closest_m < METRIC_P0_M and zone_name == "CENTER") else "P1"
            obstacles.append({
                "tier": tier,
                "label": "obstacle",
                "zone": zone_name,
                "depth_m": closest_m,
            })

    return obstacles
