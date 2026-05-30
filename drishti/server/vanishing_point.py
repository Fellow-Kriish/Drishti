"""
Vanishing point zone boundaries.

Estimates the vanishing point from the depth map horizon band and uses it
as the center anchor for zone splitting. Falls back to fixed equal thirds
if the estimate is unstable (phone tilt, turning, etc.).
"""

import numpy as np
from collections import deque

STABILITY_THRESHOLD_PX = 80    # max frame-to-frame shift before fallback
HISTORY_FRAMES         = 8     # frames to smooth over


class VanishingPointEstimator:

    def __init__(self, frame_width: int):
        self._w       = frame_width
        self._history = deque(maxlen=HISTORY_FRAMES)

    def update(self, depth_norm: np.ndarray) -> int | None:
        """
        Returns estimated vanishing point x-coordinate, or None if unstable.
        Vanishing point = horizontal position of the "most far and stable" region
        near the horizon line of the depth map.
        """
        h, w = depth_norm.shape[:2]

        # Horizon band: rows 40%–60% of frame height
        horizon_start = int(h * 0.40)
        horizon_end   = int(h * 0.60)
        horizon_band  = depth_norm[horizon_start:horizon_end, :]

        # Column means in the horizon band — the vanishing point column
        # has the lowest depth values (farthest away in disparity space)
        col_means = horizon_band.mean(axis=0)
        # In disparity space: lowest value = farthest → vanishing point
        vp_x_raw  = int(np.argmin(col_means))

        self._history.append(vp_x_raw)

        if len(self._history) < HISTORY_FRAMES:
            return None   # not enough history yet

        # Stability check: reject if estimate is jumping around
        spread = max(self._history) - min(self._history)
        if spread > STABILITY_THRESHOLD_PX:
            return None   # unstable — caller uses fixed thirds

        return int(np.mean(self._history))

    def column_boundaries(self, vp_x: int | None) -> list[int]:
        """
        Returns [0, left_boundary, right_boundary, frame_width].
        Symmetric around vanishing point if stable, equal thirds if not.
        """
        if vp_x is None:
            third = self._w // 3
            return [0, third, 2 * third, self._w]

        half_center = self._w // 6   # center zone = ±1/6 frame around vp
        left_bound  = max(0,        vp_x - half_center)
        right_bound = min(self._w,  vp_x + half_center)
        return [0, left_bound, right_bound, self._w]
