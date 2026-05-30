"""
Model loading for YOLOv8n and Depth Anything V2 Small.

Both models are loaded once at startup and held in module-level variables.
A warmup inference is run on each to trigger CUDA kernel compilation.
"""

import time
import logging

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import YOLO_MODEL_PATH, DEPTH_MODEL_ID, YOLO_CONFIDENCE

logger = logging.getLogger("drishti.models")

# ── Module-level model references ────────────────────────────────────────────
_yolo_model: YOLO | None = None
_depth_model = None
_depth_processor = None
_device: str = "cpu"


def get_device() -> str:
    """Return the active device string."""
    return _device


def load_models() -> None:
    """Load both models onto GPU (or CPU fallback) and run warmup."""
    global _yolo_model, _depth_model, _depth_processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {_device}")

    if _device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

    # ── YOLOv8n ──────────────────────────────────────────────────────────────
    logger.info(f"Loading YOLOv8n from {YOLO_MODEL_PATH}...")
    t0 = time.perf_counter()
    _yolo_model = YOLO(YOLO_MODEL_PATH)
    # Move to device — YOLO handles this via the device parameter in predict()
    logger.info(f"YOLOv8n loaded in {time.perf_counter() - t0:.2f}s")

    # ── Depth Anything V2 Small ──────────────────────────────────────────────
    logger.info(f"Loading Depth Anything V2 from {DEPTH_MODEL_ID}...")
    t0 = time.perf_counter()
    _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = _depth_model.to(_device)
    _depth_model.eval()
    logger.info(f"Depth Anything V2 loaded in {time.perf_counter() - t0:.2f}s")

    # ── Warmup ───────────────────────────────────────────────────────────────
    logger.info("Running warmup inference...")
    dummy = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    _warmup_yolo(dummy)
    _warmup_depth(dummy)
    logger.info("Warmup complete.")

    if _device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"VRAM: {alloc:.0f} MB allocated, {reserved:.0f} MB reserved")


def _warmup_yolo(img: Image.Image) -> None:
    """Run a dummy YOLO inference to compile CUDA kernels."""
    t0 = time.perf_counter()
    _yolo_model.predict(img, device=_device, verbose=False, conf=YOLO_CONFIDENCE)
    logger.info(f"YOLO warmup: {(time.perf_counter() - t0) * 1000:.1f}ms")


def _warmup_depth(img: Image.Image) -> None:
    """Run a dummy depth inference to compile CUDA kernels."""
    t0 = time.perf_counter()
    inputs = _depth_processor(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        _depth_model(**inputs)
    logger.info(f"Depth warmup: {(time.perf_counter() - t0) * 1000:.1f}ms")


# ── Inference functions ──────────────────────────────────────────────────────

def run_yolo(img: Image.Image) -> list[dict]:
    """
    Run YOLOv8n on an image.

    Returns:
        List of detections, each: {label, confidence, bbox: (x1, y1, x2, y2)}
        Coordinates are in the input image's pixel space.
    """
    results = _yolo_model.predict(
        img,
        device=_device,
        verbose=False,
        conf=YOLO_CONFIDENCE,
    )

    detections = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            label = r.names[cls_id]

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })

    return detections


def run_depth(img: Image.Image) -> np.ndarray:
    """
    Run Depth Anything V2 on an image.

    Returns:
        2D numpy array of predicted depth values, resized to input image dimensions.
        Higher values = closer (disparity map).
    """
    inputs = _depth_processor(images=img, return_tensors="pt").to(_device)

    with torch.no_grad():
        outputs = _depth_model(**inputs)

    # Post-process: resize depth to original image size
    post = _depth_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(img.height, img.width)],
    )

    depth_map = post[0]["predicted_depth"].cpu().numpy()
    return depth_map
