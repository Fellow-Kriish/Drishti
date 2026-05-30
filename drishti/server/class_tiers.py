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


def get_base_tier(class_name: str) -> str | None:
    """Return the base tier for a YOLO class, or None if unmapped."""
    return CLASS_TO_BASE_TIER.get(class_name)
