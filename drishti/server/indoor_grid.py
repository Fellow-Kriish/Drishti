"""
3×2 occupancy grid for indoor navigation.

Grid layout (after ceiling mask applied):
  Row 0 = upper half of valid frame (walls, obstacles at head/torso height)
  Row 1 = lower half of valid frame (floor-level obstacles, approaching ground)
  Col 0 = left third, Col 1 = center third, Col 2 = right third
"""

import numpy as np

from config import CEIL_MASK_FRAC, P0_THRESH, P1_THRESH, P2_THRESH


def build_occupancy_grid(normed_depth: np.ndarray) -> np.ndarray:
    """
    Returns a 2×3 array of median depth scores.
    NaN cells (ceiling) are excluded from median computation.
    Lower score = closer = more occupied.
    """
    h, w = normed_depth.shape
    ceil_cutoff = int(h * CEIL_MASK_FRAC)
    valid_h = h - ceil_cutoff
    mid_row = ceil_cutoff + valid_h // 2

    row_splits = [ceil_cutoff, mid_row, h]
    col_splits = [0, w // 3, 2 * w // 3, w]

    grid = np.full((2, 3), np.nan)
    for r in range(2):
        for c in range(3):
            cell = normed_depth[
                row_splits[r] : row_splits[r + 1],
                col_splits[c] : col_splits[c + 1],
            ]
            valid = cell[~np.isnan(cell)]
            if valid.size > 0:
                grid[r, c] = np.median(valid)
    return grid


def grid_cell_tier(score: float) -> int:
    """Returns alert tier (0=P0, 1=P1, 2=P2, 3=clear) for a grid cell score."""
    if np.isnan(score):
        return 3
    if score < P0_THRESH:
        return 0
    if score < P1_THRESH:
        return 1
    if score < P2_THRESH:
        return 2
    return 3
