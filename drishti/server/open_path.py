"""
Depth-map column sampling for open-path direction.

Samples the depth map ground plane directly. Uses METERS.
"""

import numpy as np

CLOSE_THRESHOLD_M = 2.0    # ground pixels closer than 2m = occupied
GROUND_ROWS_START = 0.55   # bottom 45% of frame = ground plane region
BBOX_MARGIN_PX    = 4      # shrink bbox slightly before masking


def open_path_direction(
    depth_meters: np.ndarray,
    detections: list,
    col_boundaries: list[int] | None = None,
    close_threshold_m: float = CLOSE_THRESHOLD_M,
) -> str:
    """
    Returns "Open path: left" / "ahead" / "right".
    Uses depth map ground-plane occupancy in meters.
    """
    h, w = depth_meters.shape[:2]

    # Ground plane region: bottom 45% of frame
    ground_start = int(h * GROUND_ROWS_START)
    ground = depth_meters[ground_start:, :]

    # Mask out pixels inside YOLO bounding boxes
    mask = np.ones_like(ground, dtype=bool)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_depth"]
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

        # In meters: pixels BELOW threshold = occupied (close)
        occupied = np.sum(valid_pixels < close_threshold_m)
        loads.append(occupied / valid_pixels.size)

    min_load = min(loads)
    if loads[1] == min_load:   return "Open path: ahead"
    if loads[0] == min_load:   return "Open path: left"
    return "Open path: right"
