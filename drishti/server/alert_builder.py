"""
Alert message builder and multi-object priority selector.

When multiple objects are detected in a single frame, only the single
highest-priority alert is sent to prevent audio queue flooding.
"""

from class_tiers import TIER_ORDER


# ── Human-readable message templates ─────────────────────────────────────────

_TIER_VERBS = {
    "P0": "Stop now",
    "P1": "Slow",
    "P2": "Caution",
    "P3": "Notice",
    "P4": "Info",
}

_ZONE_SUFFIX = {
    "LEFT": "on left",
    "CENTER": "ahead",
    "RIGHT": "on right",
}


def build_message(tier: str, label: str, zone: str) -> str:
    """
    Build a concise human-readable alert message.

    Examples:
        "Stop now, car ahead"
        "Slow, person on left"
        "Caution, bicycle on right"
    """
    verb = _TIER_VERBS.get(tier, "Alert")
    direction = _ZONE_SUFFIX.get(zone, "ahead")
    return f"{verb}, {label} {direction}"


# ── Priority selection ───────────────────────────────────────────────────────

def select_highest_priority(
    alerts: list[dict],
) -> dict | None:
    """
    Select the single highest-priority alert from a list.

    Each alert dict must have keys: tier, zone, and optionally depth_m.
    Alerts without a depth key (e.g. ground_scan, indoor alerts) are
    treated as distance 999 so they never beat a closer YOLO detection
    at the same tier.

    Priority:
        1. Highest tier (P0 > P1 > P2 > ...)
        2. Among same tier, closest depth (lowest depth_m / depth_score value)

    Returns:
        The winning alert dict, or None if the list is empty.
    """
    if not alerts:
        return None

    return min(
        alerts,
        key=lambda a: (
            TIER_ORDER.index(a["tier"]),
            a.get("depth_m", a.get("depth_score", 999)),
        ),
    )
