"""
Zone detection: determines whether a detected object is on the
LEFT, CENTER, or RIGHT of the frame, and maps that to a stereo pan channel.

Supports dynamic column boundaries from VanishingPointEstimator,
falling back to equal thirds if not provided.
"""


def get_zone(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    col_boundaries: list[int] | None = None,
) -> str:
    """
    Determine spatial zone based on bounding box center x-coordinate.

    Args:
        bbox: (x1, y1, x2, y2) in frame coordinates.
        frame_width: Width of the frame/depth map.
        col_boundaries: Optional [0, left_bound, right_bound, frame_width]
                        from VanishingPointEstimator. Falls back to equal
                        thirds if None.

    Returns:
        "LEFT", "CENTER", or "RIGHT"
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0

    if col_boundaries is not None:
        left_bound = col_boundaries[1]
        right_bound = col_boundaries[2]
    else:
        third = frame_width / 3.0
        left_bound = third
        right_bound = 2 * third

    if center_x < left_bound:
        return "LEFT"
    elif center_x < right_bound:
        return "CENTER"
    else:
        return "RIGHT"


def get_pan_channel(zone: str) -> float:
    """
    Map a zone to a stereo pan value.

    Returns:
        -1.0 (left), 0.0 (center), or 1.0 (right)
    """
    return {
        "LEFT": -1.0,
        "CENTER": 0.0,
        "RIGHT": 1.0,
    }.get(zone, 0.0)
