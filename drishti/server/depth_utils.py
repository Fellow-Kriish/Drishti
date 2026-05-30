"""
Indoor depth utilities: ceiling mask, EMA smoothing, rolling window normalization.

Processing order per frame: mask → EMA → normalize → grid.
All operations are NaN-aware to respect the ceiling mask.
"""

import numpy as np
from collections import deque

from config import CEIL_MASK_FRAC, EMA_ALPHA, ROLLING_WINDOW_N


class IndoorDepthProcessor:
    """
    Session-scoped processor for indoor depth maps.
    Maintains EMA state and rolling window history.
    """

    def __init__(self):
        self._ema_depth: np.ndarray | None = None
        self._depth_history: deque = deque(maxlen=ROLLING_WINDOW_N)

    @property
    def window_ready(self) -> bool:
        return len(self._depth_history) >= ROLLING_WINDOW_N

    def process(self, depth_map: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Full indoor depth processing: mask → EMA → rolling norm.

        Returns:
            (normalised_depth_map, window_ready)
        """
        # 1. Ceiling mask
        masked = self._apply_ceiling_mask(depth_map)

        # 2. EMA smoothing
        smoothed = self._apply_ema(masked)

        # 3. Rolling window normalization
        self._depth_history.append(smoothed)
        normed = self._normalize(smoothed)

        return normed, self.window_ready

    def _apply_ceiling_mask(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Zero out the top ceil_frac of the depth map.
        Ceiling pixels are textureless and produce unstable depth values indoors.
        Returns a copy with ceiling region set to NaN (excluded from all stats).
        """
        masked = depth_map.copy().astype(float)
        cutoff = int(depth_map.shape[0] * CEIL_MASK_FRAC)
        masked[:cutoff, :] = np.nan
        return masked

    def _apply_ema(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Exponential moving average across frames.
        alpha=0.5 at ~5.5 FPS ≈ 0.4 at 8 FPS in effective smoothing.
        NaN pixels (ceiling) are excluded — EMA only on valid pixels.
        """
        if self._ema_depth is None or self._ema_depth.shape != depth_map.shape:
            self._ema_depth = depth_map.copy()
            return self._ema_depth

        valid = ~np.isnan(depth_map)
        self._ema_depth[valid] = (
            EMA_ALPHA * depth_map[valid] + (1 - EMA_ALPHA) * self._ema_depth[valid]
        )
        self._ema_depth[~valid] = np.nan
        return self._ema_depth.copy()

    def _normalize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalise against rolling 5th/95th percentile of history.
        Falls back to per-frame normalisation during cold start.
        """
        if not self.window_ready:
            # Cold start — per-frame fallback
            scene_min = np.nanpercentile(depth_map, 5)
            scene_max = np.nanpercentile(depth_map, 95)
        else:
            all_vals = np.concatenate(
                [d[~np.isnan(d)] for d in self._depth_history]
            )
            scene_min = np.percentile(all_vals, 5)
            scene_max = np.percentile(all_vals, 95)

        denom = scene_max - scene_min
        if denom < 1e-6:
            denom = 1e-6

        normed = (depth_map - scene_min) / denom
        normed = np.clip(normed, 0.0, 1.0)
        normed[np.isnan(depth_map)] = np.nan  # preserve ceiling mask
        return normed
