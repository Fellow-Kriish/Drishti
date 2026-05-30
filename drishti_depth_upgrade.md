# Drishti — Depth Upgrade Implementation Plan
**Team Antigravity | Addendum to `drishti_hackathon_plan.md`**

---

## Overview

Six additions to the existing pipeline. No new models required. All changes are server-side Python unless noted.

Build in this order — each step depends on the previous one being correct:

1. Letterbox + coordinate remap
2. Rolling window depth normalization
3. Bounding box cross-validation fallback
4. Depth-map column sampling for open-path
5. Ground-plane discontinuity scan
6. Vanishing point zone boundaries

---

## 1. Letterbox + Coordinate Remap

**Why:** Camera is 640×480 (4:3). Stretching to 518×518 or 640×640 breaks perspective geometry. Depth Anything V2 and YOLO both degrade at distorted edges — exactly where ground-plane hazards appear.

**Rule:** All server-side resizing uses letterbox (pad, don't stretch). All coordinate math happens in letterboxed space. Only convert back to original space if drawing debug overlays on Android.

### Letterbox utility (write and test this before touching any model)

```python
# utils/letterbox.py

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


def to_letterbox_coords(x, y, meta: dict) -> tuple[float, float]:
    """Original frame coords → letterboxed coords."""
    return (
        x * meta["scale"] + meta["pad_left"],
        y * meta["scale"] + meta["pad_top"],
    )


def to_original_coords(x, y, meta: dict) -> tuple[float, float]:
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
```

### Server frame processing (replace existing resize calls)

```python
# server/pipeline.py

from utils.letterbox import letterbox, remap_bbox
from PIL import Image

def process_frame(jpeg_bytes: bytes):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    # Two separate letterbox ops — different target sizes
    yolo_img,  yolo_meta  = letterbox(img, 640)   # YOLO native
    depth_img, depth_meta = letterbox(img, 518)   # Depth Anything V2 native

    yolo_results = yolo_model(yolo_img)
    depth_map    = depth_model(depth_img)         # shape: (518, 518), float32

    detections = []
    for det in yolo_results:
        bbox_yolo  = det.bbox                                      # in 640×640 space
        bbox_depth = remap_bbox(bbox_yolo, yolo_meta, depth_meta)  # in 518×518 space
        detections.append({
            "label":      det.label,
            "bbox_depth": bbox_depth,
            "bbox_yolo":  bbox_yolo,
        })

    return depth_map, detections, depth_meta, yolo_meta
```

### Validation test (run before integration)

```python
# tests/test_letterbox.py

from utils.letterbox import letterbox, remap_bbox
from PIL import Image
import numpy as np

img = Image.new("RGB", (640, 480))
arr, meta = letterbox(img, 518)
assert arr.shape == (518, 518, 3)
assert meta["pad_top"] > 0       # 4:3 → padding on top/bottom
assert meta["pad_left"] == 0     # no horizontal padding

# Round-trip a point
x_orig, y_orig = 320.0, 240.0
from utils.letterbox import to_letterbox_coords, to_original_coords
xl, yl = to_letterbox_coords(x_orig, y_orig, meta)
xr, yr = to_original_coords(xl, yl, meta)
assert abs(xr - x_orig) < 0.5
assert abs(yr - y_orig) < 0.5

print("Letterbox tests passed.")
```

---

## 2. Rolling Window Depth Normalization

**Why:** Per-frame normalization (`frame_min`/`frame_max`) makes scores unstable. A hand entering the frame corner, a passing auto-rickshaw, a tree branch overhead — any transient intrusion reshuffles every depth score in that frame. Same object, same distance, different tier.

**Fix:** Normalize against a rolling 2-second baseline. Only update when the environment genuinely changes.

### Replace `get_depth_score` in its entirety

```python
# server/depth_normalizer.py

import numpy as np
from collections import deque

class DepthNormalizer:
    """
    Maintains a rolling window of depth map percentiles.
    Stabilizes depth scores against transient frame intrusions.
    """

    WINDOW_FRAMES = 16          # ~2 seconds at 8 FPS
    SCENE_MIN_PERCENTILE = 5    # robust floor — not absolute min
    SCENE_MAX_PERCENTILE = 95   # robust ceiling — not absolute max

    def __init__(self):
        self._mins = deque(maxlen=self.WINDOW_FRAMES)
        self._maxs = deque(maxlen=self.WINDOW_FRAMES)

    @property
    def ready(self) -> bool:
        return len(self._mins) >= self.WINDOW_FRAMES

    def update(self, depth_map: np.ndarray):
        self._mins.append(np.percentile(depth_map, self.SCENE_MIN_PERCENTILE))
        self._maxs.append(np.percentile(depth_map, self.SCENE_MAX_PERCENTILE))

    def normalize(self, depth_map: np.ndarray) -> np.ndarray:
        if not self.ready:
            # Cold-start fallback: per-frame normalization
            lo = depth_map.min()
            hi = depth_map.max()
        else:
            lo = np.mean(self._mins)
            hi = np.mean(self._maxs)

        return (depth_map - lo) / (hi - lo + 1e-6)

    def score(self, depth_map_normalized: np.ndarray, bbox: tuple) -> float:
        """
        10th percentile depth inside bbox.
        Lower score = closer to camera.
        bbox: (x1, y1, x2, y2) in depth map space.
        """
        x1, y1, x2, y2 = bbox
        # Clamp to depth map bounds
        h, w = depth_map_normalized.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        region = depth_map_normalized[y1:y2, x1:x2]
        if region.size == 0:
            return 1.0  # treat empty region as far

        return float(np.percentile(region, 10))
```

### Integration in pipeline

```python
# server/pipeline.py  (additions)

normalizer = DepthNormalizer()   # one instance, lives for the server session

def process_frame(jpeg_bytes: bytes):
    # ... letterbox, run models (from section 1) ...

    normalizer.update(depth_map)
    depth_norm = normalizer.normalize(depth_map)

    if not normalizer.ready:
        # Signal Android to play startup audio — "Drishti warming up"
        return {"tier": "P4", "message": "Drishti warming up", "pan_channel": 0.0}

    # use depth_norm for all downstream scoring
    ...
```

### Calibration procedure

Run this once in your demo environment before setting tier thresholds:

```python
# tools/calibrate_depth.py
# Walk toward a wall. Stand at known distances. Log scores.

import csv, time

LOG_FILE = "calibration_log.csv"

def log_calibration(depth_score: float, label: str):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), f"{depth_score:.4f}", label, "MEASURE_MANUALLY"])

# Call this per detection during calibration walk:
# log_calibration(score, "wall")
# Then pull the CSV and set your tier thresholds from real data.
```

---

## 3. Bounding Box Cross-Validation Fallback

**Why:** Depth Anything V2 degrades silently — low light, lens smear, uniform textures. It never raises an exception. A false "path clear" on a silent failure is the most dangerous outcome for a blind user.

**Fix:** Use YOLO bounding box area ratio as a geometric sanity check. If the two signals strongly disagree, escalate rather than trust the depth score.

### Which classes to cross-validate

Only classes with predictable real-world size. Add `"cv_eligible"` to `class_tiers.py`:

```python
# server/class_tiers.py  (additions)

# True = consistent real-world size → cross-validation valid
# False = size varies too much → skip cross-validation
CV_ELIGIBLE = {
    "person":     True,
    "bicycle":    True,
    "motorcycle": True,
    "dog":        True,
    "cow":        True,
    "car":        True,
    "truck":      False,   # huge size variance
    "bus":        False,
    "chair":      False,
    "bench":      False,
}
```

### Cross-validation logic

```python
# server/cross_validator.py

from class_tiers import CLASS_TO_BASE_TIER, CV_ELIGIBLE

# Bounding box area ratio → proximity bucket
def bbox_proximity(bbox_yolo: tuple, frame_size: int = 640) -> str:
    x1, y1, x2, y2 = bbox_yolo
    area  = (x2 - x1) * (y2 - y1)
    ratio = area / (frame_size * frame_size)

    if ratio > 0.25:  return "close"
    if ratio > 0.04:  return "medium"
    return "far"

# Depth score → proximity bucket
def depth_proximity(score: float) -> str:
    if score < 0.20:  return "close"
    if score < 0.55:  return "medium"
    return "far"

TIER_ORDER = ["P0", "P1", "P2", "P3", "P4"]

def escalate_tier(tier: str, steps: int = 1) -> str:
    idx = max(0, TIER_ORDER.index(tier) - steps)
    return TIER_ORDER[idx]

def validated_tier(
    depth_score: float,
    bbox_yolo: tuple,
    object_class: str,
    depth_tier: str,
) -> str:
    """
    Returns the final tier after cross-validation.
    Biases toward closer (more dangerous) estimate on disagreement.
    """
    if not CV_ELIGIBLE.get(object_class, False):
        return depth_tier  # skip cross-validation for this class

    bp = bbox_proximity(bbox_yolo)
    dp = depth_proximity(depth_score)

    if bp == dp:
        return depth_tier                    # agree → trust depth

    if bp == "close" and dp == "far":
        return "P0"                          # hard contradiction → maximum safety

    if bp == "close" and dp == "medium":
        return escalate_tier(depth_tier, 1)  # disagree → escalate one step

    if bp == "medium" and dp == "far":
        return escalate_tier(depth_tier, 1)  # disagree → escalate one step

    return depth_tier                        # depth says closer → already conservative
```

### Integration in pipeline

```python
# server/pipeline.py  (additions)

from cross_validator import validated_tier
from depth_normalizer import DepthNormalizer   # from section 2

for det in detections:
    score      = normalizer.score(depth_norm, det["bbox_depth"])
    raw_tier   = resolve_tier(score, det["label"])
    final_tier = validated_tier(score, det["bbox_yolo"], det["label"], raw_tier)
    # use final_tier for alert building
```

---

## 4. Depth-Map Column Sampling for Open-Path

**Why:** Current `openPathDirection()` sums bounding box area per column. A person at 20 metres in center column marks it "blocked". On a busy street, center is almost always blocked by distant detections — guidance becomes useless.

**Fix:** Sample the depth map directly. Count close-range depth pixels per column, excluding ground inside bounding boxes. The column with fewest close pixels is the open path.

```python
# server/open_path.py

import numpy as np

CLOSE_THRESHOLD   = 0.35   # depth score below this = occupied ground
GROUND_ROWS_START = 0.55   # bottom 45% of frame = ground plane region
BBOX_MARGIN_PX    = 4      # shrink bbox slightly before masking

def open_path_direction(
    depth_norm: np.ndarray,
    detections: list,
    close_threshold: float = CLOSE_THRESHOLD,
) -> str:
    """
    Returns "Open path: left" / "center" / "right".
    Uses depth map ground-plane occupancy, not bounding box area.
    """
    h, w = depth_norm.shape

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

    # Column boundaries (equal thirds — overridden by section 6 if vanishing point available)
    col_boundaries = [0, w // 3, 2 * w // 3, w]

    loads = []
    for i in range(3):
        x_start = col_boundaries[i]
        x_end   = col_boundaries[i + 1]
        col_ground = ground[:, x_start:x_end]
        col_mask   = mask[:, x_start:x_end]

        valid_pixels = col_ground[col_mask]
        if valid_pixels.size == 0:
            loads.append(0.0)
            continue

        # Fraction of valid ground pixels that are close
        occupied = np.sum(valid_pixels < close_threshold)
        loads.append(occupied / valid_pixels.size)

    min_load = min(loads)
    if loads[1] == min_load:   return "Open path: ahead"
    if loads[0] == min_load:   return "Open path: left"
    return "Open path: right"
```

### Integration

```python
# server/pipeline.py  (additions)

from open_path import open_path_direction

path_msg = open_path_direction(depth_norm, detections)
# Emit as P3 alert if no P0/P1 detected this frame
```

---

## 5. Ground-Plane Discontinuity Scan

**Why:** YOLO has no COCO class for open drains, kerbs, broken steps, or potholes. These hazards produce no bounding box. They are the primary danger on Indian streets. The depth map sees them as abrupt depth transitions in the ground plane.

**No extra model required.** Pure NumPy on the depth map already computed.

```python
# server/ground_scan.py

import numpy as np

# Tuning parameters — calibrate empirically (see calibration procedure below)
GROUND_START_FRACTION = 0.55   # scan bottom 45% of frame
GRADIENT_THRESHOLD    = 0.18   # depth change per pixel → hazard edge
CLUSTER_MIN_PIXELS    = 12     # minimum edge pixels to count as a real hazard
HAZARD_DEPTH_MAX      = 0.40   # only alert for hazards in close range

def scan_ground_plane(
    depth_norm: np.ndarray,
    detections: list,
    vanishing_point_x: int | None = None,
) -> dict | None:
    """
    Returns alert dict or None.
    alert: {"tier": str, "message": str, "pan_channel": float}
    """
    h, w = depth_norm.shape
    ground_start = int(h * GROUND_START_FRACTION)
    ground = depth_norm[ground_start:, :]

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

    # Only care about close-range hazards
    close_edges = edge_pixels & (ground < HAZARD_DEPTH_MAX)

    if np.sum(close_edges) < CLUSTER_MIN_PIXELS:
        return None

    # Find which zone the hazard is in
    edge_xs = np.where(close_edges)[1]   # x coordinates of edge pixels
    hazard_x = int(np.median(edge_xs))

    zone, pan = _zone_from_x(hazard_x, w, vanishing_point_x)

    # Tier based on how close (depth score of the hazard pixels)
    hazard_depth = float(np.median(ground[close_edges]))
    tier = "P0" if hazard_depth < 0.15 else "P1" if hazard_depth < 0.30 else "P2"

    return {
        "tier":        tier,
        "message":     f"Surface hazard, {zone}",
        "pan_channel": pan,
        "source":      "ground_scan",   # distinguish from object alerts in CSV log
    }


def _zone_from_x(x: int, frame_width: int, vp_x: int | None) -> tuple[str, float]:
    center = vp_x if vp_x is not None else frame_width // 2
    third  = frame_width // 3

    if x < center - third // 2:  return "left",   -1.0
    if x > center + third // 2:  return "right",   1.0
    return "center", 0.0
```

### Calibration procedure for ground scan

```python
# tools/calibrate_ground_scan.py
# Walk toward: (1) a known kerb, (2) a known drain edge, (3) flat clear ground.
# Log gradient magnitudes at each. Set GRADIENT_THRESHOLD above the flat-ground
# noise floor but below the kerb/drain signal.

import numpy as np, csv, time

LOG = "ground_calibration.csv"

def log_ground_scan(grad_mag: np.ndarray, label: str):
    region_max  = float(grad_mag.max())
    region_mean = float(grad_mag.mean())
    with open(LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), label, f"{region_max:.4f}", f"{region_mean:.4f}"])

# Run during a calibration walk. Collect rows for:
#   label="flat"  → sets your noise floor
#   label="kerb"  → minimum signal you must detect
#   label="drain" → minimum signal you must detect
# Set GRADIENT_THRESHOLD = flat_max * 1.5 (some headroom above noise)
```

### Integration in pipeline

```python
# server/pipeline.py  (additions)

from ground_scan import scan_ground_plane

ground_alert = scan_ground_plane(depth_norm, detections, vanishing_point_x=vp_x)

# Ground scan alert competes in the same priority queue as object alerts.
# If ground_alert tier is higher than best object alert, ground_alert wins.
all_alerts = object_alerts[:]
if ground_alert:
    all_alerts.append(ground_alert)

all_alerts.sort(key=lambda a: TIER_ORDER.index(a["tier"]))
final_alert = all_alerts[0] if all_alerts else None
```

---

## 6. Vanishing Point Zone Boundaries

**Why:** Fixed equal thirds assume the camera is perfectly centered and level. Phone tilt, road curves, and user body angle all shift where "straight ahead" actually is in the frame. Fixed thirds produce wrong left/right guidance when the phone is slightly rotated — common while walking.

**Fix:** Estimate the vanishing point from the depth map. Use it as the center anchor for zone splitting. Fall back to fixed thirds if the estimate is unstable.

```python
# server/vanishing_point.py

import numpy as np
from collections import deque

STABILITY_THRESHOLD_PX = 80    # max frame-to-frame shift before fallback
HISTORY_FRAMES         = 8     # frames to smooth over

class VanishingPointEstimator:

    def __init__(self, frame_width: int):
        self._w       = frame_width
        self._history = deque(maxlen=HISTORY_FRAMES)

    def update(self, depth_norm: np.ndarray) -> int | None:
        """
        Returns estimated vanishing point x-coordinate, or None if unstable.
        Vanishing point = horizontal position of the "most far and stable" region
        near the horizon line of the depth map.
        """
        h, w = depth_norm.shape

        # Horizon band: rows 40%–60% of frame height
        horizon_start = int(h * 0.40)
        horizon_end   = int(h * 0.60)
        horizon_band  = depth_norm[horizon_start:horizon_end, :]

        # Column means in the horizon band — the vanishing point column
        # has the highest (farthest) depth values
        col_means = horizon_band.mean(axis=0)
        vp_x_raw  = int(np.argmax(col_means))

        self._history.append(vp_x_raw)

        if len(self._history) < HISTORY_FRAMES:
            return None   # not enough history yet

        # Stability check: reject if estimate is jumping around
        spread = max(self._history) - min(self._history)
        if spread > STABILITY_THRESHOLD_PX:
            return None   # unstable — caller uses fixed thirds

        return int(np.mean(self._history))

    def column_boundaries(self, vp_x: int | None) -> list[int]:
        """
        Returns [0, left_boundary, right_boundary, frame_width].
        Symmetric around vanishing point if stable, equal thirds if not.
        """
        if vp_x is None:
            third = self._w // 3
            return [0, third, 2 * third, self._w]

        half_center = self._w // 6   # center zone = ±1/6 frame around vp
        left_bound  = max(0,        vp_x - half_center)
        right_bound = min(self._w,  vp_x + half_center)
        return [0, left_bound, right_bound, self._w]
```

### Integration in pipeline

```python
# server/pipeline.py  (additions)

from vanishing_point import VanishingPointEstimator

vp_estimator = VanishingPointEstimator(frame_width=518)   # one instance per session

def process_frame(jpeg_bytes: bytes):
    # ... letterbox, models, normalize (sections 1–2) ...

    vp_x        = vp_estimator.update(depth_norm)
    col_bounds  = vp_estimator.column_boundaries(vp_x)

    # Pass col_bounds into open_path_direction and ground scan
    path_msg    = open_path_direction(depth_norm, detections, col_bounds)
    ground_alert = scan_ground_plane(depth_norm, detections, vanishing_point_x=vp_x)
    ...
```

### Update `open_path_direction` signature

```python
# server/open_path.py  (updated signature)

def open_path_direction(
    depth_norm: np.ndarray,
    detections: list,
    col_boundaries: list[int] | None = None,   # from VanishingPointEstimator
    close_threshold: float = CLOSE_THRESHOLD,
) -> str:
    h, w = depth_norm.shape
    boundaries = col_boundaries or [0, w // 3, 2 * w // 3, w]
    # rest of function unchanged — uses boundaries instead of hardcoded thirds
    ...
```

---

## Full Server Pipeline (assembled)

```python
# server/pipeline.py  — complete frame handler

import io
import numpy as np
from PIL import Image

from utils.letterbox      import letterbox, remap_bbox
from depth_normalizer     import DepthNormalizer
from cross_validator      import validated_tier
from open_path            import open_path_direction
from ground_scan          import scan_ground_plane
from vanishing_point      import VanishingPointEstimator
from class_tiers          import CLASS_TO_BASE_TIER, TIER_ORDER
from depth_normalizer     import DepthNormalizer

# Module-level singletons (live for server session)
normalizer   = DepthNormalizer()
vp_estimator = VanishingPointEstimator(frame_width=518)

last_alert_time = {}   # for deduplication — carries over from existing plan


def process_frame(jpeg_bytes: bytes) -> dict | None:
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

    # 1. Letterbox
    yolo_img,  yolo_meta  = letterbox(img, 640)
    depth_img, depth_meta = letterbox(img, 518)

    # 2. Run models
    yolo_results = yolo_model(yolo_img)
    depth_map    = depth_model(depth_img)

    # 3. Rolling normalization
    normalizer.update(depth_map)
    if not normalizer.ready:
        return {"tier": "P4", "message": "Drishti warming up", "pan_channel": 0.0}
    depth_norm = normalizer.normalize(depth_map)

    # 4. Vanishing point
    vp_x       = vp_estimator.update(depth_norm)
    col_bounds = vp_estimator.column_boundaries(vp_x)

    # 5. Build detections with remapped bboxes
    detections = []
    for det in yolo_results:
        bbox_yolo  = det.bbox
        bbox_depth = remap_bbox(bbox_yolo, yolo_meta, depth_meta)
        score      = normalizer.score(depth_norm, bbox_depth)
        raw_tier   = resolve_tier(score, det.label)
        if raw_tier is None:
            continue
        final_tier = validated_tier(score, bbox_yolo, det.label, raw_tier)
        zone, pan  = zone_from_col_bounds(bbox_depth, depth_meta["target_size"], col_bounds)
        detections.append({
            "label":      det.label,
            "bbox_depth": bbox_depth,
            "bbox_yolo":  bbox_yolo,
            "tier":       final_tier,
            "score":      score,
            "zone":       zone,
            "pan":        pan,
        })

    # 6. Ground plane scan
    ground_alert = scan_ground_plane(depth_norm, detections, vanishing_point_x=vp_x)

    # 7. Open path direction (P3 guidance)
    path_msg = open_path_direction(depth_norm, detections, col_bounds)

    # 8. Merge all alerts, send highest priority
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

## File Structure (additions only)

```
server/
├── utils/
│   └── letterbox.py          # Section 1
├── depth_normalizer.py       # Section 2
├── cross_validator.py        # Section 3
├── open_path.py              # Section 4
├── ground_scan.py            # Section 5
├── vanishing_point.py        # Section 6
├── pipeline.py               # assembled handler
├── class_tiers.py            # existing — add CV_ELIGIBLE dict
└── tools/
    ├── calibrate_depth.py    # Section 2 calibration
    └── calibrate_ground_scan.py  # Section 5 calibration
```

---

## Calibration Checklist (run before demo)

| Step | What to do | What to set |
|---|---|---|
| GPU validation | Run 100 frames, check CUDA, watch VRAM | Confirm < 3GB VRAM used |
| Depth thresholds | Walk toward wall at 0.5m / 1m / 2m / 4m, log scores | Set P0/P1/P2/P3 score boundaries in `resolve_tier` |
| Ground scan | Walk toward kerb + drain, log gradient magnitudes | Set `GRADIENT_THRESHOLD` in `ground_scan.py` |
| Rolling window | Watch score stability on a busy scene for 30 seconds | Confirm no tier flipping on stationary objects |
| Vanishing point | Walk straight, log `vp_x` — should stay near 259 (518/2) | Confirm `STABILITY_THRESHOLD_PX` is appropriate |
| Cross-validation | Trigger intentional depth failure (cover lens briefly) | Confirm escalation fires, not suppression |

---

## Latency Impact of Additions

| Addition | Extra compute | Notes |
|---|---|---|
| Letterbox | < 2ms | PIL resize + paste |
| Rolling normalization | < 1ms | deque + percentile on small buffer |
| Cross-validation | < 1ms | arithmetic on existing numbers |
| Depth column sampling | < 3ms | NumPy slice + percentile × 3 |
| Ground scan | < 5ms | `np.gradient` on 518×518 subregion |
| Vanishing point | < 2ms | column mean on horizon band |
| **Total added** | **< 14ms** | Well within 400ms P0 budget |

---

*Addendum to `drishti_hackathon_plan.md` · Team Antigravity · Drishti v0.2*
