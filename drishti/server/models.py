"""
Model loading for YOLOv8n and Depth Anything V2 Metric (Small).

Both depth models (indoor/outdoor) are loaded at startup.
Active model is swapped on mode toggle — no reload delay.
"""

import sys
import time
import logging

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Add official Depth Anything V2 metric_depth to Python path
sys.path.insert(0, "Depth-Anything-V2/metric_depth")
from depth_anything_v2.dpt import DepthAnythingV2

from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE

logger = logging.getLogger("drishti.models")

# ── Module-level model references ────────────────────────────────────────────
_yolo_model: YOLO | None = None
_depth_model_indoor = None
_depth_model_outdoor = None
_depth_model = None          # active model pointer — swapped by mode toggle
_device: str = "cpu"

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


def get_device() -> str:
    """Return the active device string."""
    return _device


def _load_depth_model(dataset: str):
    """
    Load a Depth Anything V2 metric model.
    dataset = "hypersim" → indoor (max_depth=20m)
    dataset = "vkitti"   → outdoor (max_depth=80m)
    """
    max_depth = 20 if dataset == "hypersim" else 80
    model = DepthAnythingV2(**{**MODEL_CONFIGS["vits"], "max_depth": max_depth})
    ckpt_path = f"checkpoints/depth_anything_v2_metric_{dataset}_vits.pth"
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")
    )
    return model.to(_device).eval()


def load_models() -> None:
    """Load all models onto GPU (or CPU fallback) and run warmup."""
    global _yolo_model, _depth_model_indoor, _depth_model_outdoor, _depth_model, _device

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
    logger.info(f"YOLOv8n loaded in {time.perf_counter() - t0:.2f}s")

    # ── Depth Anything V2 Metric (both checkpoints) ─────────────────────────
    logger.info("Loading Depth Anything V2 Metric — Indoor (hypersim)...")
    t0 = time.perf_counter()
    _depth_model_indoor = _load_depth_model("hypersim")
    logger.info(f"Indoor depth model loaded in {time.perf_counter() - t0:.2f}s")

    logger.info("Loading Depth Anything V2 Metric — Outdoor (vkitti)...")
    t0 = time.perf_counter()
    _depth_model_outdoor = _load_depth_model("vkitti")
    logger.info(f"Outdoor depth model loaded in {time.perf_counter() - t0:.2f}s")

    # Default to indoor
    _depth_model = _depth_model_indoor

    # ── Warmup ───────────────────────────────────────────────────────────────
    logger.info("Running warmup inference...")
    dummy_pil = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    _warmup_yolo(dummy_pil)
    _warmup_depth(dummy_bgr)
    logger.info("Warmup complete.")

    if _device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(f"VRAM: {alloc:.0f} MB allocated, {reserved:.0f} MB reserved")


def swap_depth_model(mode: str) -> None:
    """Swap the active depth model based on mode toggle."""
    global _depth_model
    if mode == "indoor":
        _depth_model = _depth_model_indoor
    else:
        _depth_model = _depth_model_outdoor
    logger.info(f"Depth model swapped to: {mode}")


def _warmup_yolo(img: Image.Image) -> None:
    """Run a dummy YOLO inference to compile CUDA kernels."""
    t0 = time.perf_counter()
    _yolo_model.predict(img, device=_device, verbose=False, conf=YOLO_CONFIDENCE)
    logger.info(f"YOLO warmup: {(time.perf_counter() - t0) * 1000:.1f}ms")


def _warmup_depth(frame_bgr: np.ndarray) -> None:
    """Run a dummy depth inference to compile CUDA kernels."""
    t0 = time.perf_counter()
    _depth_model.infer_image(frame_bgr)
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


def run_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Run Depth Anything V2 Metric on a BGR image.

    Returns:
        2D numpy array of predicted depth values in METERS.
        Lower values = closer.
    """
    return _depth_model.infer_image(frame_bgr)
