"""
Letterbox utility for aspect-ratio-preserving resizing.

Camera is 640×480 (4:3). Stretching to 518×518 or 640×640 breaks perspective
geometry. This module ensures all resizing uses letterbox (pad, don't stretch).
"""

import numpy as np
from PIL import Image


def letterbox(img: Image.Image, target_size: int) -> tuple[np.ndarray, dict]:
    """
    Resize image preserving aspect ratio, pad with black to target_size × target_size.
    Returns (letterboxed_array, meta) where meta is needed for coordinate remap.
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    pad_left = (target_size - new_w) // 2
    pad_top  = (target_size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "orig_w": w,
        "orig_h": h,
        "target_size": target_size,
    }
    return np.array(canvas), meta


def to_letterbox_coords(x: float, y: float, meta: dict) -> tuple[float, float]:
    """Original frame coords → letterboxed coords."""
    return (
        x * meta["scale"] + meta["pad_left"],
        y * meta["scale"] + meta["pad_top"],
    )


def to_original_coords(x: float, y: float, meta: dict) -> tuple[float, float]:
    """Letterboxed coords → original frame coords."""
    return (
        (x - meta["pad_left"]) / meta["scale"],
        (y - meta["pad_top"])  / meta["scale"],
    )


def remap_bbox(bbox: tuple, from_meta: dict, to_meta: dict) -> tuple:
    """
    Remap bbox from one letterboxed space to another.
    Use this to map YOLO 640×640 boxes → Depth 518×518 space.
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    # to original
    x1o, y1o = to_original_coords(x1, y1, from_meta)
    x2o, y2o = to_original_coords(x2, y2, from_meta)
    # to target letterbox
    x1t, y1t = to_letterbox_coords(x1o, y1o, to_meta)
    x2t, y2t = to_letterbox_coords(x2o, y2o, to_meta)
    return (int(x1t), int(y1t), int(x2t), int(y2t))
