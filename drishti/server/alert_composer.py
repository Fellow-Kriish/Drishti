"""
Indoor alert composition — priority queue for indoor navigation alerts.

Priority order (highest to lowest, first match wins):
  stairs > wall_closing > step > obstacle > wall > doorway > surface_warning

Enforces suppression window between identical alerts to avoid TTS flooding.
"""

import numpy as np
from time import monotonic

from config import ALERT_SUPPRESS_S
from indoor_grid import grid_cell_tier

FRAME_WIDTH = 640

PRIORITY_ORDER = [
    "stairs",
    "wall_closing",
    "step",
    "obstacle",
    "wall",
    "doorway",
    "surface_warning",
]

# Pan channel mapping
_PAN_MAP = {"left": -1.0, "center": 0.0, "right": 1.0}

# Tier string mapping
_TIER_STR = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}


class IndoorAlertComposer:
    """Session-scoped alert composer with suppression state."""

    def __init__(self):
        self._last_alert_time = 0.0
        self._last_alert_text = ""

    def compose(
        self,
        corridor_info: dict,
        ground_hazard: dict | None,
        doorway: dict | None,
        yolo_confirmed: list[dict],
        grid: np.ndarray,
        mirror_anomaly: bool,
        window_ready: bool,
    ) -> dict:
        """
        Returns the single highest-priority alert as a JSON-serialisable dict.
        Enforces ALERT_SUPPRESS_S suppression window between identical alerts.
        """
        now = monotonic()
        alerts = []

        # 1. Stairs (P0 always)
        if ground_hazard and ground_hazard["type"] == "stairs":
            alerts.append({
                "priority": 0,
                "text": "Stairs ahead, stop",
                "tier": 0,
                "pan": "center",
            })

        # 2. Wall closing in — ONLY in confirmed narrow_corridor AND center is actually close
        #    narrowing trend alone is not enough; mode must be narrow_corridor
        if (
            corridor_info["mode"] == "narrow_corridor"
            and corridor_info["narrowing"]
            and corridor_info["center_tier"] <= 1
        ):
            alerts.append({
                "priority": 1,
                "text": "Path narrowing",
                "tier": corridor_info["center_tier"],
                "pan": "center",
            })

        # 3. Step
        if ground_hazard and ground_hazard["type"] == "step":
            alerts.append({
                "priority": 2,
                "text": f"Step ahead {ground_hazard['zone']}",
                "tier": ground_hazard["tier"],
                "pan": ground_hazard["zone"],
            })

        # 4. YOLO obstacle (confirmed, non-mirror)
        if not mirror_anomaly:
            for det in yolo_confirmed:
                bbox = det["bbox"]
                cx = (bbox[0] + bbox[2]) // 2
                zone = (
                    "left" if cx < FRAME_WIDTH // 3
                    else ("right" if cx > 2 * FRAME_WIDTH // 3 else "center")
                )
                col = 0 if zone == "left" else (2 if zone == "right" else 1)
                tier = grid_cell_tier(grid[0, col])
                alerts.append({
                    "priority": 3,
                    "text": f"{det['label']} {zone}",
                    "tier": tier,
                    "pan": zone,
                })

        # 5. Wall alert (single wall, not narrow corridor)
        if corridor_info["mode"] == "wall_left":
            alerts.append({
                "priority": 4,
                "text": "Wall on left",
                "tier": 1,
                "pan": "left",
            })
        elif corridor_info["mode"] == "wall_right":
            alerts.append({
                "priority": 4,
                "text": "Wall on right",
                "tier": 1,
                "pan": "right",
            })

        # 6. Doorway (informational)
        if doorway:
            alerts.append({
                "priority": 5,
                "text": f"Opening {doorway['zone']}",
                "tier": 3,
                "pan": doorway["zone"],
            })

        # 7. Reflective surface
        if mirror_anomaly:
            alerts.append({
                "priority": 6,
                "text": "Reflective surface",
                "tier": 2,
                "pan": "center",
            })

        if not alerts:
            return {"alert": False, "warming_up": not window_ready}

        best = min(alerts, key=lambda a: a["priority"])

        # Suppression: same text within window → skip
        if (
            best["text"] == self._last_alert_text
            and (now - self._last_alert_time) < ALERT_SUPPRESS_S
        ):
            return {"alert": False, "warming_up": not window_ready}

        self._last_alert_time = now
        self._last_alert_text = best["text"]

        return {
            "alert": True,
            "tier": _TIER_STR.get(best["tier"], "P3"),
            "message": best["text"],
            "pan_channel": _PAN_MAP.get(best["pan"], 0.0),
            "warming_up": not window_ready,
        }
