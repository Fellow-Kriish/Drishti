"""
Indoor zone analysis: corridor detection, doorway detection, mirror anomaly detection.

Replaces vanishing-point zone logic entirely for indoor mode.
"""

import numpy as np

from config import CEIL_MASK_FRAC, P1_THRESH
from indoor_grid import grid_cell_tier

# ── Corridor detection ───────────────────────────────────────────────────────

NARROW_CORRIDOR_THRESH = 0.35          # both walls must be VERY close to count as narrow corridor
NARROW_CORRIDOR_FRAMES = 8             # must persist 8 frames (~1.5s) to confirm

DOORWAY_FAR_THRESH = 0.65   # depth score — a "far" region in otherwise close frame
DOORWAY_MIN_WIDTH  = 60     # minimum pixel width of the gap (~10% of 640)

MIRROR_DEPTH_MIN = 0.75     # depth model thinks this region is very far


class IndoorZoneAnalyser:
    """Session-scoped indoor zone analyser with temporal state."""

    def __init__(self):
        self._narrow_counter = 0
        self._center_history: list[float] = []
        self._center_trend_window = 5

    def analyse_corridor(self, grid: np.ndarray) -> dict:
        """
        Analyses the 2×3 grid and returns zone guidance.

        Returns dict:
            mode:        "narrow_corridor" | "wall_left" | "wall_right" | "open_room"
            center_tier: int (tier of center column — the walking path)
            wall_left:   float (left column upper-half median score)
            wall_right:  float (right column upper-half median score)
            narrowing:   bool (center column score dropping = path closing in)
        """
        left_score = grid[0, 0]    # upper-left  = left wall
        center_score = grid[0, 1]  # upper-center = path ahead
        right_score = grid[0, 2]   # upper-right = right wall

        left_close = (not np.isnan(left_score)) and left_score < NARROW_CORRIDOR_THRESH
        right_close = (not np.isnan(right_score)) and right_score < NARROW_CORRIDOR_THRESH

        if left_close and right_close:
            self._narrow_counter += 1
        else:
            self._narrow_counter = max(0, self._narrow_counter - 1)

        narrow = self._narrow_counter >= NARROW_CORRIDOR_FRAMES

        return {
            "mode": (
                "narrow_corridor" if narrow else
                ("wall_left" if left_close else
                 ("wall_right" if right_close else "open_room"))
            ),
            "center_tier": grid_cell_tier(center_score),
            "wall_left": float(left_score) if not np.isnan(left_score) else 1.0,
            "wall_right": float(right_score) if not np.isnan(right_score) else 1.0,
            "narrowing": False,  # populated by update_center_trend below
        }

    def update_center_trend(self, center_score: float) -> bool:
        """
        Returns True if center depth score is meaningfully and consistently
        dropping — i.e. the path is genuinely closing in.

        Guards against noise-level fluctuations by requiring:
          1. Monotonically decreasing over the window
          2. Total drop > CENTER_TREND_MIN_DELTA (avoids tiny noise triggering)
          3. Mode must be narrow_corridor (checked in alert_composer)
        """
        CENTER_TREND_MIN_DELTA = 0.12  # must drop at least 0.12 across the window

        if np.isnan(center_score):
            return False

        self._center_history.append(center_score)
        if len(self._center_history) > self._center_trend_window:
            self._center_history.pop(0)
        if len(self._center_history) < self._center_trend_window:
            return False

        # Must be monotonically decreasing
        monotonic = all(
            self._center_history[i] > self._center_history[i + 1]
            for i in range(len(self._center_history) - 1)
        )
        if not monotonic:
            return False

        # Must also have a meaningful total drop, not just noise
        total_drop = self._center_history[0] - self._center_history[-1]
        return total_drop >= CENTER_TREND_MIN_DELTA


# ── Doorway detection ────────────────────────────────────────────────────────

def detect_doorway(normed_depth: np.ndarray, corridor_info: dict) -> dict | None:
    """
    Looks for a vertical strip of far depth in the upper half of the frame.
    Only meaningful in corridor modes (not open_room).
    Returns dict with zone, or None.
    """
    if corridor_info["mode"] == "open_room":
        return None

    h, w = normed_depth.shape
    ceil_cutoff = int(h * CEIL_MASK_FRAC)
    upper_half = normed_depth[ceil_cutoff : ceil_cutoff + (h - ceil_cutoff) // 2, :]

    # Column-wise median of upper half
    col_medians = np.nanmedian(upper_half, axis=0)

    # Find columns that are "far" (potential doorway gap)
    far_cols = np.where(col_medians > DOORWAY_FAR_THRESH)[0]

    if len(far_cols) < DOORWAY_MIN_WIDTH:
        return None

    # Find the widest contiguous run of far columns
    gaps = np.split(far_cols, np.where(np.diff(far_cols) > 5)[0] + 1)
    widest = max(gaps, key=len)

    center_col = int(np.mean(widest))
    zone = "left" if center_col < w // 3 else ("right" if center_col > 2 * w // 3 else "center")

    return {"zone": zone, "width_px": len(widest)}


# ── Mirror / reflective surface detection ────────────────────────────────────

def check_mirror_anomaly(
    yolo_detections: list[dict],
    grid: np.ndarray,
    corridor_info: dict,
) -> bool:
    """
    Returns True if a YOLO detection is in a region the depth map says is far —
    inside a corridor where that distance is physically impossible.
    """
    if corridor_info["mode"] == "open_room":
        return False

    for det in yolo_detections:
        bbox = det["bbox"]
        box_center_x = (bbox[0] + bbox[2]) // 2
        # Map to grid column
        w = 640
        col = 0 if box_center_x < w // 3 else (2 if box_center_x > 2 * w // 3 else 1)
        cell_score = grid[0, col]
        if not np.isnan(cell_score) and cell_score > MIRROR_DEPTH_MIN:
            return True  # YOLO sees object where depth says nothing is there
    return False
