"""
Depth-map column sampling for open-path direction.

Samples the depth map ground plane directly instead of summing bounding box
area per column. A person at 20m in center column no longer marks it "blocked".
"""

import numpy as np

CLOSE_THRESHOLD   = 0.35   # depth score below this = occupied ground
GROUND_ROWS_START = 0.55   # bottom 45% of frame = ground plane region
BBOX_MARGIN_PX    = 4      # shrink bbox slightly before masking


def open_path_direction(
    depth_norm: np.ndarray,
    detections: list,
    col_boundaries: list[int] | None = None,
    close_threshold: float = CLOSE_THRESHOLD,
) -> str:
    """
    Returns "Open path: left" / "ahead" / "right".
    Uses depth map ground-plane occupancy, not bounding box area.
    """
    h, w = depth_norm.shape[:2]

    # Ground plane region: bottom 45% of frame
    ground_start = int(h * GROUND_ROWS_START)
    ground = depth_norm[ground_start:, :]   # shape: (rows, w)

    # Mask out pixels inside YOLO bounding boxes (object surfaces, not ground)
    mask = np.ones_like(ground, dtype=bool)  # True = valid ground pixel
    for det in detections:
        x1, y1, x2, y2 = det["bbox_depth"]
        # Remap y to ground-plane slice coordinates
        y1g = max(0, y1 - ground_start - BBOX_MARGIN_PX)
        y2g = max(0, y2 - ground_start + BBOX_MARGIN_PX)
        x1c = max(0, x1 - BBOX_MARGIN_PX)
        x2c = min(w,  x2 + BBOX_MARGIN_PX)
        mask[y1g:y2g, x1c:x2c] = False

    # Column boundaries (from vanishing point or equal thirds fallback)
    boundaries = col_boundaries or [0, w // 3, 2 * w // 3, w]

    loads = []
    for i in range(3):
        x_start = boundaries[i]
        x_end   = boundaries[i + 1]
        col_ground = ground[:, x_start:x_end]
        col_mask   = mask[:, x_start:x_end]

        valid_pixels = col_ground[col_mask]
        if valid_pixels.size == 0:
            loads.append(0.0)
            continue

        # For disparity-based depth (higher = closer after normalization):
        # "close" pixels are those with HIGH normalized values
        # But after our DepthNormalizer, we still have disparity space
        # So high values = close. We count pixels ABOVE threshold as occupied.
        occupied = np.sum(valid_pixels > (1.0 - close_threshold))
        loads.append(occupied / valid_pixels.size)

    min_load = min(loads)
    if loads[1] == min_load:   return "Open path: ahead"
    if loads[0] == min_load:   return "Open path: left"
    return "Open path: right"
