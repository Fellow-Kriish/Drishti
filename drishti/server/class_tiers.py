"""
YOLO COCO class → base alert tier mapping.

Define this table BEFORE writing any other server logic.
Objects not in this table produce no alert.
"""

# Tier priority order — lower index = higher priority
TIER_ORDER = ["P0", "P1", "P2", "P3", "P4"]

CLASS_TO_BASE_TIER: dict[str, str] = {
    # ── P0 — immediate stop ──────────────────────────────────────────────────
    "car": "P0",
    "truck": "P0",
    "bus": "P0",
    "motorcycle": "P0",
    # COCO doesn't have auto_rickshaw — alias from "motorcycle" or fine-tune
    # "auto_rickshaw": "P0",

    # ── P1 — slow down ───────────────────────────────────────────────────────
    "person": "P1",
    "dog": "P1",
    "cow": "P1",
    "bicycle": "P1",

    # ── P2 — warning ─────────────────────────────────────────────────────────
    "cat": "P2",
    "chair": "P2",
    "bench": "P2",
    "potted plant": "P2",
    "dining table": "P2",
    "bed": "P2",
    "sofa": "P2",
    "tv": "P2",
    "laptop": "P2",
    "cell phone": "P2",
    "book": "P2",

    # ── P3 — guidance ─────────────────────────────────────────────────────────
    "traffic light": "P3",
    "stop sign": "P3",
}


# ── Cross-validation eligibility ─────────────────────────────────────────────
# True = consistent real-world size → cross-validation valid
# False = size varies too much → skip cross-validation
CV_ELIGIBLE: dict[str, bool] = {
    "person":     True,
    "bicycle":    True,
    "motorcycle": True,
    "dog":        True,
    "cow":        True,
    "car":        True,
    "truck":      False,   # huge size variance
    "bus":        False,
    "chair":      False,
    "bench":      False,
}


def get_base_tier(class_name: str) -> str | None:
    """Return the base tier for a YOLO class, or None if unmapped."""
    return CLASS_TO_BASE_TIER.get(class_name)


# ── Dynamic obstacle classification ──────────────────────────────────────────
# Objects in this set are NEVER suppressed in corridor mode.
# Objects not in this set (and not detected by YOLO) = Static Structure → suppressible.
DYNAMIC_CLASSES: set[str] = {
    # Moving hazards
    "person",
    "dog",
    "cow",
    "bicycle",
    "motorcycle",
    "car",
    "truck",
    "bus",
    # Furniture — static physically, but critical trip hazards in corridors.
    # A chair in a narrow corridor must always alert.
    "chair",
    "dining table",
    "bench",
    "potted plant",
}


def is_dynamic(class_name: str) -> bool:
    """
    Returns True if the object class should never be suppressed in corridor mode.
    Returns False for pure depth-map detections (walls, floor textures, ceiling artifacts).
    """
    return class_name in DYNAMIC_CLASSES
