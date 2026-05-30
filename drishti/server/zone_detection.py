"""
Zone detection: determines whether a detected object is on the
LEFT, CENTER, or RIGHT of the frame, and maps that to a stereo pan channel.
"""


def get_zone(bbox: tuple[int, int, int, int], frame_width: int) -> str:
    """
    Determine spatial zone based on bounding box center x-coordinate.

    Args:
        bbox: (x1, y1, x2, y2) in frame coordinates.
        frame_width: Width of the frame/depth map.

    Returns:
        "LEFT", "CENTER", or "RIGHT"
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0

    third = frame_width / 3.0

    if center_x < third:
        return "LEFT"
    elif center_x < 2 * third:
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
