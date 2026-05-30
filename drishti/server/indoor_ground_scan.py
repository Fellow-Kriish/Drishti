"""
Indoor ground-plane discontinuity scan for steps and stairs.

Scans the bottom GROUND_ROI_FRAC of the frame for depth gradient spikes.
Temporal persistence distinguishes single steps (P1) from stairs (P0).
"""

import numpy as np
from collections import deque

from config import GROUND_ROI_FRAC

# Tuning parameters — calibrate from real step/stair walks
GROUND_GRAD_THRESH = 0.18   # gradient magnitude spike threshold
GROUND_CLUSTER_MIN = 15     # minimum pixels in a discontinuity cluster
STAIR_FRAME_COUNT  = 3      # consecutive frames with same-band discontinuity = stairs


class IndoorGroundScanner:
    """Session-scoped ground scanner with temporal state for stair detection."""

    def __init__(self):
        self._discontinuity_history: deque = deque(maxlen=STAIR_FRAME_COUNT)

    def scan(
        self,
        normed_depth: np.ndarray,
        yolo_detections: list[dict],
    ) -> dict | None:
        """
        Scans the bottom GROUND_ROI_FRAC of the frame for depth discontinuities.
        Excludes pixels inside YOLO bounding boxes (to avoid object edges).

        Returns dict with keys: type ("step"|"stairs"), zone, tier (int)
        Returns None if no discontinuity found.
        """
        h, w = normed_depth.shape
        roi_start = int(h * (1 - GROUND_ROI_FRAC))
        ground = normed_depth[roi_start:, :].copy()

        # Mask out YOLO box regions in ground ROI
        for det in yolo_detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            # Remap to ground ROI coordinates
            y1_roi = max(0, y1 - roi_start)
            y2_roi = max(0, y2 - roi_start)
            if y2_roi > 0:
                ground[y1_roi:y2_roi, x1:x2] = np.nan

        # Compute gradient magnitude
        gy, gx = np.gradient(np.nan_to_num(ground, nan=0.0))
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Threshold
        spike_mask = grad_mag > GROUND_GRAD_THRESH
        spike_count = int(np.sum(spike_mask))

        if spike_count < GROUND_CLUSTER_MIN:
            self._discontinuity_history.append(None)
            return None

        # Find dominant zone of the discontinuity
        spike_rows, spike_cols = np.where(spike_mask)
        median_col = int(np.median(spike_cols))
        zone = (
            "left" if median_col < w // 3
            else ("right" if median_col > 2 * w // 3 else "center")
        )

        self._discontinuity_history.append(zone)

        # Stairs: same zone fires in all recent frames
        if (
            len(self._discontinuity_history) == STAIR_FRAME_COUNT
            and all(z == zone for z in self._discontinuity_history)
        ):
            hazard_type = "stairs"
            tier = 0  # P0 — always stop for stairs
        else:
            hazard_type = "step"
            tier = 1  # P1

        return {"type": hazard_type, "zone": zone, "tier": tier}
