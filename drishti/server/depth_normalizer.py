"""
Rolling window depth normalization.

Normalizes depth maps against a 2-second rolling baseline instead of
per-frame min/max. This stabilizes depth scores against transient
frame intrusions (hands, passing vehicles, overhead branches).
"""

import numpy as np
from collections import deque


class DepthNormalizer:
    """
    Maintains a rolling window of depth map percentiles.
    Stabilizes depth scores against transient frame intrusions.
    """

    WINDOW_FRAMES = 16          # ~2 seconds at 8 FPS
    SCENE_MIN_PERCENTILE = 5    # robust floor — not absolute min
    SCENE_MAX_PERCENTILE = 95   # robust ceiling — not absolute max

    def __init__(self):
        self._mins = deque(maxlen=self.WINDOW_FRAMES)
        self._maxs = deque(maxlen=self.WINDOW_FRAMES)

    @property
    def ready(self) -> bool:
        return len(self._mins) >= self.WINDOW_FRAMES

    def update(self, depth_map: np.ndarray):
        self._mins.append(np.percentile(depth_map, self.SCENE_MIN_PERCENTILE))
        self._maxs.append(np.percentile(depth_map, self.SCENE_MAX_PERCENTILE))

    def normalize(self, depth_map: np.ndarray) -> np.ndarray:
        if not self.ready:
            # Cold-start fallback: per-frame normalization
            lo = depth_map.min()
            hi = depth_map.max()
        else:
            lo = np.mean(self._mins)
            hi = np.mean(self._maxs)

        return (depth_map - lo) / (hi - lo + 1e-6)

    def score(self, depth_map_normalized: np.ndarray, bbox: tuple) -> float:
        """
        10th percentile depth inside bbox.
        Lower score = closer to camera.
        bbox: (x1, y1, x2, y2) in depth map space.

        Note: Depth Anything outputs disparity (higher = closer).
        After normalization, higher values = closer.
        We invert so that lower score = closer (consistent with existing tier logic).
        """
        x1, y1, x2, y2 = bbox
        # Clamp to depth map bounds
        h, w = depth_map_normalized.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 1.0  # treat empty region as far

        region = depth_map_normalized[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0  # treat empty region as far

        # 90th percentile = closest surface (highest disparity after normalization)
        closest_surface = float(np.percentile(region, 90))

        # Invert: 0.0 = closest, 1.0 = farthest
        # After normalization, values near 1.0 are the closest (highest disparity)
        # We want lower score = closer, so invert
        return float(1.0 - closest_surface)
