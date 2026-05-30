"""
3×2 occupancy grid for indoor navigation.

Grid layout (after ceiling mask applied):
  Row 0 = upper half of valid frame (walls, obstacles at head/torso height)
  Row 1 = lower half of valid frame (floor-level obstacles, approaching ground)
  Col 0 = left third, Col 1 = center third, Col 2 = right third

All values are in METERS (metric depth).
"""

import numpy as np

from config import CEIL_MASK_FRAC, P0_THRESH_M, P1_THRESH_M, P2_THRESH_M


def build_occupancy_grid(depth_meters: np.ndarray) -> np.ndarray:
    """
    Returns a 2×3 array of median depth in meters.
    NaN cells (ceiling) are excluded from median computation.
    Lower score = closer = more occupied.
    """
    h, w = depth_meters.shape
    ceil_cutoff = int(h * CEIL_MASK_FRAC)
    valid_h = h - ceil_cutoff
    mid_row = ceil_cutoff + valid_h // 2

    row_splits = [ceil_cutoff, mid_row, h]
    col_splits = [0, w // 3, 2 * w // 3, w]

    grid = np.full((2, 3), np.nan)
    for r in range(2):
        for c in range(3):
            cell = depth_meters[
                row_splits[r] : row_splits[r + 1],
                col_splits[c] : col_splits[c + 1],
            ]
            valid = cell[~np.isnan(cell)]
            if valid.size > 0:
                grid[r, c] = np.median(valid)
    return grid


def grid_cell_tier(score_meters: float) -> int:
    """Returns alert tier (0=P0, 1=P1, 2=P2, 3=clear) for a grid cell score in meters."""
    if np.isnan(score_meters):
        return 3
    if score_meters < P0_THRESH_M:
        return 0
    if score_meters < P1_THRESH_M:
        return 1
    if score_meters < P2_THRESH_M:
        return 2
    return 3
