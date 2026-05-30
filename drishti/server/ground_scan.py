"""
Ground-plane discontinuity scan (outdoor).

Detects kerbs, drains, potholes, broken steps — hazards that produce no
YOLO bounding box but appear as abrupt depth transitions on the ground plane.

All depth values are in METERS (metric depth).
"""

import numpy as np

# Tuning parameters
GROUND_START_FRACTION = 0.55   # scan bottom 45% of frame
GRADIENT_THRESHOLD    = 0.18   # depth change per pixel → hazard edge (unitless)
CLUSTER_MIN_PIXELS    = 12     # minimum edge pixels to count as a real hazard
HAZARD_DEPTH_MAX_M    = 3.0    # only alert for hazards within 3 meters


def scan_ground_plane(
    depth_meters: np.ndarray,
    detections: list,
    vanishing_point_x: int | None = None,
) -> dict | None:
    """
    Returns alert dict or None.
    alert: {"tier": str, "message": str, "pan_channel": float, "source": str}
    """
    h, w = depth_meters.shape[:2]
    ground_start = int(h * GROUND_START_FRACTION)
    ground = depth_meters[ground_start:, :]

    # Mask out bounding box regions (object surfaces, not ground)
    mask = np.ones_like(ground, dtype=bool)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_depth"]
        y1g = max(0, y1 - ground_start)
        y2g = max(0, y2 - ground_start)
        mask[y1g:y2g, x1:x2] = False

    # Gradient magnitude on ground region
    gy, gx = np.gradient(ground)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Apply mask and threshold
    edge_pixels = (grad_mag > GRADIENT_THRESHOLD) & mask

    if not np.any(edge_pixels):
        return None

    # Only care about close-range hazards (meters: lower = closer)
    close_edges = edge_pixels & (ground < HAZARD_DEPTH_MAX_M)

    if np.sum(close_edges) < CLUSTER_MIN_PIXELS:
        return None

    # Find which zone the hazard is in
    edge_xs = np.where(close_edges)[1]
    hazard_x = int(np.median(edge_xs))

    zone, pan = _zone_from_x(hazard_x, w, vanishing_point_x)

    # Tier based on depth in meters
    hazard_depth_m = float(np.median(ground[close_edges]))
    tier = "P0" if hazard_depth_m < 0.8 else "P1" if hazard_depth_m < 1.5 else "P2"

    return {
        "tier":        tier,
        "message":     f"Surface hazard, {zone}",
        "pan_channel": pan,
        "source":      "ground_scan",
        "depth_m":     hazard_depth_m,
    }


def _zone_from_x(x: int, frame_width: int, vp_x: int | None) -> tuple[str, float]:
    center = vp_x if vp_x is not None else frame_width // 2
    third  = frame_width // 3

    if x < center - third // 2:  return "left",   -1.0
    if x > center + third // 2:  return "right",   1.0
    return "center", 0.0
