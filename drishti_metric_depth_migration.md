# Drishti — Metric Depth Migration
**Addendum to `drishti_hackathon_plan.md`, `drishti_depth_upgrade.md`, `drishti_indoor_pipeline.md`**
**Execute in order. Each step is independently testable before moving to the next.**

---

## What Changes

| Before | After |
|---|---|
| HuggingFace `transformers` pipeline | Official `.pth` + `DepthAnythingV2` class |
| Relative disparity (0.0–1.0) | Absolute meters (0.0–20.0 indoor / 0.0–80.0 outdoor) |
| `DepthNormalizer` + rolling window | Deleted |
| EMA temporal smoothing | Deleted |
| Percentile-based `resolve_tier()` | Meter-based `resolve_tier_metric()` |
| Single model for all environments | `hypersim` (indoor) / `vkitti` (outdoor) swapped on mode toggle |

---

## Step 1 — Repo Setup

Clone the official repo alongside your server:

```bash
cd your_server_directory
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth
pip install -r requirements.txt
```

Download both Small checkpoints into a `checkpoints/` folder:

```bash
mkdir -p checkpoints

# Indoor model (Hypersim, max_depth=20m)
wget -O checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true"

# Outdoor model (Virtual KITTI 2, max_depth=80m)
wget -O checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth?download=true"
```

Validate GPU before loading any model:

```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # must show RTX 2050
print(f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM")
```

---

## Step 2 — Replace Model Loading in `pipeline.py`

Remove the `transformers` import entirely. Replace with:

```python
# pipeline.py — top of file

import sys
import torch
sys.path.insert(0, "Depth-Anything-V2/metric_depth")
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}

def load_depth_model(dataset: str) -> DepthAnythingV2:
    """
    dataset = "hypersim" for indoor (max 20m)
    dataset = "vkitti"   for outdoor (max 80m)
    """
    max_depth = 20 if dataset == "hypersim" else 80
    model = DepthAnythingV2(**{**MODEL_CONFIGS["vits"], "max_depth": max_depth})
    model.load_state_dict(
        torch.load(
            f"checkpoints/depth_anything_v2_metric_{dataset}_vits.pth",
            map_location="cpu",
        )
    )
    return model.to(DEVICE).eval()

# Load both at startup — swap on mode toggle, no reload delay mid-session
depth_model_indoor  = load_depth_model("hypersim")
depth_model_outdoor = load_depth_model("vkitti")

# Active model pointer — swapped by mode toggle handler
depth_model = depth_model_indoor   # default: indoor
```

Mode toggle handler (in your existing WebSocket message handler):

```python
if msg.get("type") == "mode":
    global depth_model, current_mode
    current_mode = msg["mode"]
    depth_model  = depth_model_indoor if current_mode == "indoor" else depth_model_outdoor
    return
```

---

## Step 3 — Fix Frame Ingestion

`infer_image()` expects a **BGR numpy array** (exactly what `cv2.imdecode` gives). Do not convert to RGB. Do not resize before passing — the model handles that internally at 518×518.

```python
# pipeline.py — inside process_frame()

import cv2
import numpy as np

def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Returns BGR numpy array. Do NOT convert to RGB."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("imdecode failed — malformed JPEG")
    return frame   # shape: (H, W, 3), dtype uint8, BGR

def run_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Returns HxW depth map in meters (float32)."""
    with torch.no_grad():
        depth = depth_model.infer_image(frame_bgr)   # handles resize internally
    return depth   # values in meters
```

Validation test — run this once before integrating into the pipeline:

```python
import cv2
frame = cv2.imread("test_image.jpg")          # any image
depth = run_depth(frame)
print(f"shape: {depth.shape}")                # should be (H, W)
print(f"min: {depth.min():.2f}m  max: {depth.max():.2f}m")
# Indoor scene: expect min ~0.3m, max ~5–10m
# If min=0.0 and max=0.0 → model loaded wrong checkpoint (likely relative, not metric)
# If min=0.0 and max=20.0 flat → imdecode failed, frame is black
```

---

## Step 4 — Delete Dead Code

Remove these entirely from `depth_utils.py`:

```
# DELETE the following classes and functions:
- class DepthNormalizer          (rolling window, percentile tracking, cold-start)
- apply_ema()                    (EMA smoothing)
- update_rolling_window()        (rolling window append)
- get_normalised_depth()         (rolling normalization logic)
- depth_history deque            (module-level state)
- ema_depth global variable      (module-level state)
- window_ready global variable   (module-level state)
```

Remove these from `pipeline.py`:

```
# DELETE:
- normalizer = DepthNormalizer()         (singleton)
- normalizer.update(depth_map)           (every frame call)
- normalizer.normalize(depth_map)        (every frame call)
- the "warming_up" cold-start JSON block
- "warming_up" field from all JSON responses
```

Remove from Android (`MainActivity.kt`):

```kotlin
// DELETE:
// Any logic that plays "Drishti warming up" on warming_up == true
// The warming_up field check from JSON parsing
```

---

## Step 5 — Replace `resolve_tier` with Meter-Based Version

Delete the old `resolve_tier()` in `pipeline.py`. Replace with:

```python
# pipeline.py

TIER_ORDER = ["P0", "P1", "P2", "P3", "P4"]

# Meter thresholds — calibrate these from your corridor walk (Step 7)
# These starting values are conservative; adjust down if alerts fire too early
METRIC_THRESHOLDS = {
    "P0": 1.0,   # < 1.0m  → STOP
    "P1": 2.0,   # < 2.0m  → Slow down
    "P2": 4.0,   # < 4.0m  → Warning
    "P3": 6.0,   # < 6.0m  → Guidance
}

def resolve_tier_metric(depth_meters: float, object_class: str) -> str | None:
    """
    depth_meters: 10th percentile depth inside the object bounding box.
    Returns tier string or None (suppress alert).
    """
    base_tier = CLASS_TO_BASE_TIER.get(object_class)
    if base_tier is None:
        return None

    # Determine tier from absolute distance
    if depth_meters < METRIC_THRESHOLDS["P0"]: distance_tier = "P0"
    elif depth_meters < METRIC_THRESHOLDS["P1"]: distance_tier = "P1"
    elif depth_meters < METRIC_THRESHOLDS["P2"]: distance_tier = "P2"
    elif depth_meters < METRIC_THRESHOLDS["P3"]: distance_tier = "P3"
    else:
        return None  # too far — suppress

    # Take the more dangerous of class-based tier and distance-based tier
    class_idx    = TIER_ORDER.index(base_tier)
    distance_idx = TIER_ORDER.index(distance_tier)
    return TIER_ORDER[min(class_idx, distance_idx)]
```

Replace `get_depth_score()` — it no longer needs normalization:

```python
def get_depth_score_metric(depth_map: np.ndarray, bbox: tuple) -> float:
    """
    Returns 10th percentile depth in meters inside bbox.
    bbox: (x1, y1, x2, y2) in depth map coordinates.
    Lower = closer, in real meters.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 999.0   # treat empty region as very far
    return float(np.percentile(region, 10))
```

---

## Step 6 — Update Indoor Pipeline (`depth_utils.py`)

The indoor pipeline in `drishti_indoor_pipeline.md` used normalized 0.0–1.0 scores. Update its thresholds to meters.

In `indoor_grid.py`, replace:

```python
# OLD (normalized scores)
P0_THRESH = 0.25
P1_THRESH = 0.50
P2_THRESH = 0.70

def grid_cell_tier(score: float) -> int:
    if np.isnan(score):   return 3
    if score < P0_THRESH: return 0
    if score < P1_THRESH: return 1
    if score < P2_THRESH: return 2
    return 3
```

```python
# NEW (meters — calibrate from corridor walk)
P0_THRESH_M = 1.0   # < 1.0m
P1_THRESH_M = 2.5   # < 2.5m
P2_THRESH_M = 4.0   # < 4.0m

def grid_cell_tier(score_meters: float) -> int:
    if np.isnan(score_meters):       return 3
    if score_meters < P0_THRESH_M:   return 0
    if score_meters < P1_THRESH_M:   return 1
    if score_meters < P2_THRESH_M:   return 2
    return 3
```

In `ground_scan.py`, replace:

```python
# OLD
GROUND_GRAD_THRESH = 0.18
HAZARD_DEPTH_MAX   = 0.40

# tier logic
tier = "P0" if hazard_depth < 0.15 else "P1" if hazard_depth < 0.30 else "P2"
```

```python
# NEW
GROUND_GRAD_THRESH  = 0.18       # gradient threshold unchanged — still unitless pixel gradient
HAZARD_DEPTH_MAX_M  = 3.0        # only alert for hazards within 3 meters

# tier logic (meters)
tier = "P0" if hazard_depth < 0.8 else "P1" if hazard_depth < 1.5 else "P2"
```

In `indoor_zones.py`, replace all normalized score comparisons:

```python
# OLD
NARROW_CORRIDOR_THRESH = P1_THRESH       # 0.50
DOORWAY_FAR_THRESH     = 0.65
MIRROR_DEPTH_MIN       = 0.75

# NEW (meters)
NARROW_CORRIDOR_THRESH_M = 2.5    # wall within 2.5m = corridor wall
DOORWAY_FAR_THRESH_M     = 5.0    # opening reads as > 5m = doorway gap
MIRROR_DEPTH_MIN_M       = 8.0    # depth > 8m in a room = physically impossible = mirror
```

In `open_path.py`, replace:

```python
# OLD
CLOSE_THRESHOLD = 0.35   # normalized score

# NEW
CLOSE_THRESHOLD_M = 2.0  # ground pixels closer than 2m = occupied
```

In `vanishing_point.py` — no changes needed. It works on raw depth map geometry, not threshold values.

---

## Step 7 — Calibration Walk (Required Before Demo)

Run this once in your test environment. Log to CSV, set thresholds from real data.

Add temporarily to `pipeline.py`:

```python
import csv, time

CALIBRATION_MODE = True   # set False after calibration

def log_calibration(label: str, depth_meters: float, bbox_area_ratio: float):
    if not CALIBRATION_MODE:
        return
    with open("calibration_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            label,
            f"{depth_meters:.3f}",
            f"{bbox_area_ratio:.4f}",
            "MEASURE_MANUALLY",   # fill this column while walking
        ])
```

Call it per detection in `process_frame()`:

```python
area_ratio = ((x2 - x1) * (y2 - y1)) / (depth_map.shape[1] * depth_map.shape[0])
log_calibration(label, depth_score_meters, area_ratio)
```

Walk toward these targets and record real distance manually:

| Target | Distances to test | Threshold to set |
|---|---|---|
| Wall (straight) | 0.5m / 1.0m / 2.0m / 3.0m / 5.0m | `P0_THRESH_M`, `P1_THRESH_M`, `P2_THRESH_M` |
| Person standing | 0.5m / 1.0m / 2.0m / 3.0m | `METRIC_THRESHOLDS` P0/P1/P2 |
| Step / kerb | 1.0m / 0.5m | `HAZARD_DEPTH_MAX_M` |
| Open doorway | 2.0m / 1.0m | `DOORWAY_FAR_THRESH_M` |

After collecting data, update `config.py` with calibrated values. Set `CALIBRATION_MODE = False`.

---

## Step 8 — Updated `process_frame()` (Full Integration)

```python
# pipeline.py — complete process_frame() after migration

import io, cv2, numpy as np
from PIL import Image
from utils.letterbox import letterbox, remap_bbox
from class_tiers     import CLASS_TO_BASE_TIER, TIER_ORDER
from open_path       import open_path_direction
from ground_scan     import scan_ground_plane
from vanishing_point import VanishingPointEstimator
from cross_validator import validated_tier_metric   # see note below

vp_estimator = VanishingPointEstimator(frame_width=518)

def process_frame(jpeg_bytes: bytes) -> dict | None:
    # 1. Decode frame
    frame_bgr = decode_frame(jpeg_bytes)

    # 2. Run YOLO (letterboxed to 640×640)
    pil_img          = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    yolo_img, yolo_meta = letterbox(pil_img, 640)
    yolo_results     = yolo_model(yolo_img)

    # 3. Run Depth Anything V2 metric (pass raw BGR directly)
    depth_map = run_depth(frame_bgr)   # HxW float32 in meters

    # 4. Letterbox meta for depth space coordinate mapping
    _, depth_meta = letterbox(pil_img, 518)

    # 5. Vanishing point
    vp_x       = vp_estimator.update(depth_map)
    col_bounds = vp_estimator.column_boundaries(vp_x)

    # 6. Build detections
    detections = []
    for det in yolo_results:
        bbox_yolo  = det.bbox
        bbox_depth = remap_bbox(bbox_yolo, yolo_meta, depth_meta)
        depth_m    = get_depth_score_metric(depth_map, bbox_depth)
        tier       = resolve_tier_metric(depth_m, det.label)
        if tier is None:
            continue
        cx   = (bbox_depth[0] + bbox_depth[2]) // 2
        zone = "left" if cx < 518 // 3 else ("right" if cx > 2 * 518 // 3 else "center")
        pan  = -1.0 if zone == "left" else (1.0 if zone == "right" else 0.0)
        detections.append({
            "label":      det.label,
            "bbox_depth": bbox_depth,
            "bbox_yolo":  bbox_yolo,
            "tier":       tier,
            "depth_m":    depth_m,
            "zone":       zone,
            "pan":        pan,
        })

    # 7. Ground scan (depth_map now in meters — thresholds updated in Step 6)
    ground_alert = scan_ground_plane(depth_map, detections, vanishing_point_x=vp_x)

    # 8. Open path
    path_msg = open_path_direction(depth_map, detections, col_bounds)

    # 9. Merge and send highest priority
    all_alerts = [
        {"tier": d["tier"], "message": f"{d['label']}, {d['zone']}", "pan_channel": d["pan"]}
        for d in detections
    ]
    if ground_alert:
        all_alerts.append(ground_alert)
    if not all_alerts:
        all_alerts.append({"tier": "P3", "message": path_msg, "pan_channel": 0.0})

    all_alerts.sort(key=lambda a: TIER_ORDER.index(a["tier"]))
    return all_alerts[0]
```

---

## Step 9 — Cross-Validator Update (Metric Version)

The old `validated_tier()` used normalized depth scores. Update it to work with meters:

```python
# cross_validator.py — replace validated_tier() with this

def bbox_proximity_metric(bbox_yolo: tuple, depth_m: float) -> str:
    """Geometric sanity check: do bbox size and measured depth agree?"""
    x1, y1, x2, y2 = bbox_yolo
    area_ratio = ((x2 - x1) * (y2 - y1)) / (640 * 640)

    # Approximate expected depth from bbox area (calibrate these from your data)
    if area_ratio > 0.20:   bbox_says = "close"    # large box = should be < 2m
    elif area_ratio > 0.04: bbox_says = "medium"   # medium box = 2–5m
    else:                   bbox_says = "far"       # small box = > 5m

    if depth_m < 2.0:       depth_says = "close"
    elif depth_m < 5.0:     depth_says = "medium"
    else:                   depth_says = "far"

    return bbox_says, depth_says

def validated_tier_metric(depth_m: float, bbox_yolo: tuple,
                           object_class: str, depth_tier: str) -> str:
    if not CV_ELIGIBLE.get(object_class, False):
        return depth_tier

    bbox_says, depth_says = bbox_proximity_metric(bbox_yolo, depth_m)

    if bbox_says == depth_says:
        return depth_tier                         # agree → trust depth

    if bbox_says == "close" and depth_says == "far":
        return "P0"                               # hard contradiction → maximum safety

    if bbox_says == "close" and depth_says == "medium":
        idx = TIER_ORDER.index(depth_tier)
        return TIER_ORDER[max(0, idx - 1)]        # escalate one step

    if bbox_says == "medium" and depth_says == "far":
        idx = TIER_ORDER.index(depth_tier)
        return TIER_ORDER[max(0, idx - 1)]        # escalate one step

    return depth_tier
```

---

## Step 10 (Stretch) — ONNX Export for Speed

Only attempt this after Steps 1–9 are confirmed working and calibrated.

```bash
# Clone TensorRT/ONNX export repo
git clone https://github.com/Stillwtm/depth-anything-tensorrt
cd depth-anything-tensorrt
git submodule init && git submodule update
pip install tensorrt==10.2.0.post1

# IMPORTANT: replace dpt.py before export
cp tools/dpt.py third_party/depth_anything_v2/depth_anything_v2/dpt.py

# Export indoor metric model to ONNX
python tools/export_onnx.py \
  --checkpoint ../checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  --onnx models/depth_metric_indoor.onnx \
  --input_size 518 \
  --encoder vits \
  --batch 1 \
  --metric \
  --max_depth 20

# Export outdoor metric model to ONNX
python tools/export_onnx.py \
  --checkpoint ../checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
  --onnx models/depth_metric_outdoor.onnx \
  --input_size 518 \
  --encoder vits \
  --batch 1 \
  --metric \
  --max_depth 80
```

Validate ONNX before converting to TRT:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/depth_metric_indoor.onnx",
                             providers=["CUDAExecutionProvider"])
dummy = np.random.randn(1, 3, 518, 518).astype(np.float32)
out   = sess.run(None, {"image": dummy})
print(out[0].shape)   # should be (1, 518, 518) or (518, 518)
print(f"min: {out[0].min():.2f}  max: {out[0].max():.2f}")
# max should be close to 20.0 (not 1.0)
```

Convert to TensorRT FP16 only after ONNX validation passes:

```bash
python tools/onnx2trt.py \
  --onnx models/depth_metric_indoor.onnx \
  --engine models/depth_metric_indoor.engine \
  --fp16

python tools/onnx2trt.py \
  --onnx models/depth_metric_outdoor.onnx \
  --engine models/depth_metric_outdoor.engine \
  --fp16
```

TRT inference wrapper (replaces `run_depth()` in pipeline):

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTDepthModel:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Preprocess: BGR → RGB, resize to 518×518, normalize
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        inp  = img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 518, 518)

        # Run TRT
        inp_tensor = np.ascontiguousarray(inp)
        out_tensor = np.empty((1, 518, 518), dtype=np.float32)
        self.context.execute_v2([inp_tensor.ctypes.data, out_tensor.ctypes.data])
        return out_tensor[0]   # (518, 518) in meters

# Load at startup
trt_indoor  = TRTDepthModel("models/depth_metric_indoor.engine")
trt_outdoor = TRTDepthModel("models/depth_metric_outdoor.engine")
```

**Note:** TRT `.engine` files are GPU-specific. Rebuild on the demo machine. Do not commit engine files to the repo.

---

## File Change Summary

| File | Action |
|---|---|
| `pipeline.py` | Replace model loading, `resolve_tier`, `get_depth_score`, `process_frame` |
| `depth_utils.py` | Delete `DepthNormalizer`, `apply_ema`, `update_rolling_window`, `get_normalised_depth` |
| `indoor_grid.py` | Update `P0_THRESH`, `P1_THRESH`, `P2_THRESH` to meter values |
| `ground_scan.py` | Update `HAZARD_DEPTH_MAX` to meter value |
| `indoor_zones.py` | Update `NARROW_CORRIDOR_THRESH`, `DOORWAY_FAR_THRESH`, `MIRROR_DEPTH_MIN` to meters |
| `open_path.py` | Update `CLOSE_THRESHOLD` to meters |
| `cross_validator.py` | Replace `validated_tier()` with `validated_tier_metric()` |
| `config.py` | All calibrated thresholds live here after calibration walk |
| `MainActivity.kt` | Delete `warming_up` JSON field handling |
| `checkpoints/` | Add `depth_anything_v2_metric_hypersim_vits.pth` and `depth_anything_v2_metric_vkitti_vits.pth` |

---

*Addendum to `drishti_hackathon_plan.md` · Team Antigravity · Drishti v0.3*
