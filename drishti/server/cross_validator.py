"""
Bounding box cross-validation fallback.

Uses YOLO bounding box area ratio as a geometric sanity check against
depth scores. If the two signals strongly disagree, escalates to the
closer (more dangerous) estimate rather than trusting the depth score.
"""

from class_tiers import TIER_ORDER

# CV_ELIGIBLE is imported from class_tiers — added in this upgrade

# Bounding box area ratio → proximity bucket
def bbox_proximity(bbox_yolo: tuple, frame_size: int = 640) -> str:
    x1, y1, x2, y2 = bbox_yolo
    area  = (x2 - x1) * (y2 - y1)
    ratio = area / (frame_size * frame_size)

    if ratio > 0.25:  return "close"
    if ratio > 0.04:  return "medium"
    return "far"


# Depth score → proximity bucket
def depth_proximity(score: float) -> str:
    if score < 0.20:  return "close"
    if score < 0.55:  return "medium"
    return "far"


def escalate_tier(tier: str, steps: int = 1) -> str:
    idx = max(0, TIER_ORDER.index(tier) - steps)
    return TIER_ORDER[idx]


def validated_tier(
    depth_score: float,
    bbox_yolo: tuple,
    object_class: str,
    depth_tier: str,
) -> str:
    """
    Returns the final tier after cross-validation.
    Biases toward closer (more dangerous) estimate on disagreement.
    """
    from class_tiers import CV_ELIGIBLE

    if not CV_ELIGIBLE.get(object_class, False):
        return depth_tier  # skip cross-validation for this class

    bp = bbox_proximity(bbox_yolo)
    dp = depth_proximity(depth_score)

    if bp == dp:
        return depth_tier                    # agree → trust depth

    if bp == "close" and dp == "far":
        return "P0"                          # hard contradiction → maximum safety

    if bp == "close" and dp == "medium":
        return escalate_tier(depth_tier, 1)  # disagree → escalate one step

    if bp == "medium" and dp == "far":
        return escalate_tier(depth_tier, 1)  # disagree → escalate one step

    return depth_tier                        # depth says closer → already conservative
