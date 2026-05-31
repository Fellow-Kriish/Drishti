"""
Indoor alert composition — priority queue for indoor navigation alerts.

Priority order (highest to lowest, first match wins):
  stairs > wall_closing > step > obstacle > wall > doorway > surface_warning

Enforces suppression window between identical alerts to avoid TTS flooding.
"""

import numpy as np
from time import monotonic
import time as _time

from config import ALERT_SUPPRESS_S, P2_THRESH_M, P3_THRESH_M, P3_SUPPRESS_S
from indoor_grid import grid_cell_tier
from depth_processing import resolve_p0_p1
from class_tiers import is_dynamic

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
        looming_results: dict,
        last_p3_alert_time: float,
        last_p3_alert_text: str,
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
                zone_key = zone.upper()   # "LEFT", "CENTER", "RIGHT"
                col = 0 if zone == "left" else (2 if zone == "right" else 1)
                cell_depth_m = grid[0, col]   # metres

                # ── Looming check (P0/P1 override) ─────────────────────────────────
                looming = looming_results.get(zone_key, {})
                if looming:
                    p0p1_tier = resolve_p0_p1(
                        median_depth=looming["median_depth"],
                        raw_depth=cell_depth_m,
                        rate_of_approach=looming["rate_of_approach"],
                        zone=zone_key,
                    )
                    if p0p1_tier is not None:
                        alerts.append({
                            "priority": 3,
                            "text": f"{det['label']} {zone}",
                            "tier": {"P0": 0, "P1": 1}.get(p0p1_tier, 2),
                            "pan": zone,
                        })
                        continue   # skip standard tier check for this detection

                # ── P2 corridor suppression ───────────────────────────────────────
                in_corridor = corridor_info["mode"] in ("narrow_corridor", "wall_left", "wall_right")
                if in_corridor and not is_dynamic(det["label"]):
                    continue   # suppress static structures in corridor mode

                # ── Standard tier from grid ───────────────────────────────────────
                tier = grid_cell_tier(cell_depth_m)
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
        best_tier_str = _TIER_STR.get(best["tier"], "P3")

        # ── P3 text+time gating ──────────────────────────────────────────────
        if best_tier_str == "P3":
            elapsed = _time.monotonic() - last_p3_alert_time
            if elapsed < P3_SUPPRESS_S and best["text"] == last_p3_alert_text:
                return {"alert": False, "warming_up": not window_ready}
            # Different P3 text fires immediately — no suppression

        # ── Standard suppression (P0/P1/P2) ──────────────────────────────────
        if (
            best["text"] == self._last_alert_text
            and (now - self._last_alert_time) < ALERT_SUPPRESS_S
            and best_tier_str != "P3"   # P3 handled above
        ):
            return {"alert": False, "warming_up": not window_ready}

        self._last_alert_time = now
        self._last_alert_text = best["text"]

        return {
            "alert": True,
            "tier": best_tier_str,
            "message": best["text"],
            "pan_channel": _PAN_MAP.get(best["pan"], 0.0),
            "warming_up": not window_ready,
        }
