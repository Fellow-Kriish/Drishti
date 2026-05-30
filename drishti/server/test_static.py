"""
Drishti Server — Static Image Test

Validates the full inference pipeline without needing the Android client:
  1. Checks CUDA availability
  2. Loads both models
  3. Processes a test image (or generates a dummy one)
  4. Prints all detections with depth scores, tiers, zones
  5. Measures per-model inference time

Usage:
    python test_static.py                    # uses built-in dummy image
    python test_static.py path/to/image.jpg  # uses your image
"""

import sys
import time

import torch
import numpy as np
from PIL import Image

from config import YOLO_INPUT_SIZE, DEPTH_INPUT_SIZE
from models import load_models, run_yolo, run_depth, get_device
from depth_processing import get_depth_score, resolve_tier
from zone_detection import get_zone, get_pan_channel
from alert_builder import build_message, select_highest_priority


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_cuda() -> None:
    print_header("CUDA Check")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM: {props.total_memory / (1024**3):.1f} GB")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠  CUDA not available — running on CPU (will be slow)")


def load_test_image(path: str | None) -> Image.Image:
    if path:
        print(f"\n  Loading image: {path}")
        return Image.open(path).convert("RGB")
    else:
        print("\n  No image provided — generating dummy 640x480 image")
        # Create a simple test image with colored rectangles
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:300, 200:400] = [255, 128, 0]  # orange rectangle
        img[50:150, 50:150] = [0, 128, 255]    # blue rectangle
        return Image.fromarray(img)


def run_benchmark(img: Image.Image, n_runs: int = 10) -> None:
    print_header(f"Benchmark ({n_runs} runs)")

    yolo_img = img.resize(YOLO_INPUT_SIZE)
    depth_img = img.resize(DEPTH_INPUT_SIZE)

    # Warm up (already done in load_models, but be safe)
    run_yolo(yolo_img)
    run_depth(depth_img)

    yolo_times = []
    depth_times = []

    for i in range(n_runs):
        t0 = time.perf_counter()
        run_yolo(yolo_img)
        yolo_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        run_depth(depth_img)
        depth_times.append((time.perf_counter() - t0) * 1000)

    print(f"  YOLO:  avg {np.mean(yolo_times):.1f}ms  "
          f"(min {np.min(yolo_times):.1f}, max {np.max(yolo_times):.1f})")
    print(f"  Depth: avg {np.mean(depth_times):.1f}ms  "
          f"(min {np.min(depth_times):.1f}, max {np.max(depth_times):.1f})")
    print(f"  Combined: avg {np.mean(yolo_times) + np.mean(depth_times):.1f}ms")


def run_pipeline(img: Image.Image) -> None:
    print_header("Full Pipeline Test")

    yolo_img = img.resize(YOLO_INPUT_SIZE)
    depth_img = img.resize(DEPTH_INPUT_SIZE)

    # Run YOLO
    t0 = time.perf_counter()
    detections = run_yolo(yolo_img)
    t_yolo = (time.perf_counter() - t0) * 1000
    print(f"\n  YOLO inference: {t_yolo:.1f}ms")
    print(f"  Detections found: {len(detections)}")

    # Run Depth
    t0 = time.perf_counter()
    depth_map = run_depth(depth_img)
    t_depth = (time.perf_counter() - t0) * 1000
    print(f"  Depth inference: {t_depth:.1f}ms")
    print(f"  Depth map shape: {depth_map.shape}")
    print(f"  Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # Process each detection
    alerts = []
    depth_h, depth_w = depth_map.shape[:2]

    print(f"\n  {'Label':<15} {'Conf':>5} {'Depth':>6} {'Base':>5} {'Final':>5} {'Zone':>7} {'Pan':>5}")
    print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*5}")

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        bx1, by1, bx2, by2 = det["bbox"]

        # Scale bbox from YOLO space to depth space
        sx = depth_w / YOLO_INPUT_SIZE[0]
        sy = depth_h / YOLO_INPUT_SIZE[1]
        scaled_bbox = (int(bx1 * sx), int(by1 * sy), int(bx2 * sx), int(by2 * sy))

        depth_score = get_depth_score(depth_map, scaled_bbox)

        from class_tiers import CLASS_TO_BASE_TIER
        base_tier = CLASS_TO_BASE_TIER.get(label, "—")
        tier = resolve_tier(depth_score, label)

        zone = get_zone(scaled_bbox, depth_w)
        pan = get_pan_channel(zone)

        tier_str = tier if tier else "SKIP"
        print(f"  {label:<15} {conf:>5.2f} {depth_score:>6.3f} {base_tier:>5} {tier_str:>5} {zone:>7} {pan:>5.1f}")

        if tier:
            message = build_message(tier, label, zone)
            alerts.append({
                "tier": tier,
                "label": label,
                "message": message,
                "pan_channel": pan,
                "depth_score": depth_score,
                "zone": zone,
            })

    # Select highest priority
    best = select_highest_priority(alerts)

    print_header("Final Alert (sent to client)")
    if best:
        print(f"  tier: {best['tier']}")
        print(f"  message: \"{best['message']}\"")
        print(f"  pan_channel: {best['pan_channel']}")
        print(f"  (from: {best['label']}, depth={best['depth_score']:.3f}, zone={best['zone']})")
    else:
        print("  No alert — all objects either unmapped, too far, or no detections.")

    # VRAM usage
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"\n  VRAM: {alloc:.0f} MB allocated / {reserved:.0f} MB reserved")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    check_cuda()

    print_header("Loading Models")
    load_models()
    print(f"  Device: {get_device()}")

    img = load_test_image(image_path)

    run_pipeline(img)
    run_benchmark(img)

    print_header("✓ All tests passed")
    print()


if __name__ == "__main__":
    main()
