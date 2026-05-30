"""
Indoor depth utilities: ceiling mask only.

With metric depth, EMA smoothing and rolling normalization are no longer needed.
The model outputs absolute meters — no normalization required.
"""

import numpy as np

from config import CEIL_MASK_FRAC


class IndoorDepthProcessor:
    """
    Session-scoped processor for indoor depth maps.
    With metric depth, only ceiling masking is needed.
    """

    def process(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Apply ceiling mask to metric depth map.
        Returns depth map with ceiling region set to NaN.
        """
        return self._apply_ceiling_mask(depth_map)

    def _apply_ceiling_mask(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Set top ceil_frac of the depth map to NaN.
        Ceiling pixels are textureless and produce unstable depth values indoors.
        """
        masked = depth_map.copy().astype(float)
        cutoff = int(depth_map.shape[0] * CEIL_MASK_FRAC)
        masked[:cutoff, :] = np.nan
        return masked
