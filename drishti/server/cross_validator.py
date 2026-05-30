"""
Bounding box cross-validation fallback (metric version).

Uses YOLO bounding box area ratio as a geometric sanity check against
depth in meters. If the two signals strongly disagree, escalates to the
closer (more dangerous) estimate.
"""

from class_tiers import TIER_ORDER, CV_ELIGIBLE


def bbox_proximity(bbox_yolo: tuple, frame_size: int = 640) -> str:
    """Bounding box area ratio → proximity bucket."""
    x1, y1, x2, y2 = bbox_yolo
    area  = (x2 - x1) * (y2 - y1)
    ratio = area / (frame_size * frame_size)

    if ratio > 0.20:  return "close"    # large box = should be < 2m
    if ratio > 0.04:  return "medium"   # medium box = 2–5m
    return "far"                         # small box = > 5m


def depth_proximity_metric(depth_m: float) -> str:
    """Depth in meters → proximity bucket."""
    if depth_m < 2.0:  return "close"
    if depth_m < 5.0:  return "medium"
    return "far"


def escalate_tier(tier: str, steps: int = 1) -> str:
    idx = max(0, TIER_ORDER.index(tier) - steps)
    return TIER_ORDER[idx]


def validated_tier(
    depth_m: float,
    bbox_yolo: tuple,
    object_class: str,
    depth_tier: str,
) -> str:
    """
    Returns the final tier after cross-validation.
    Biases toward closer (more dangerous) estimate on disagreement.
    """
    if not CV_ELIGIBLE.get(object_class, False):
        return depth_tier

    bp = bbox_proximity(bbox_yolo)
    dp = depth_proximity_metric(depth_m)

    if bp == dp:
        return depth_tier                    # agree → trust depth

    if bp == "close" and dp == "far":
        return "P0"                          # hard contradiction → maximum safety

    if bp == "close" and dp == "medium":
        return escalate_tier(depth_tier, 1)

    if bp == "medium" and dp == "far":
        return escalate_tier(depth_tier, 1)

    return depth_tier
