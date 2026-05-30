# Drishti — Indoor Pipeline Implementation Plan
**Status:** Addendum to `drishti_hackathon_plan.md` and `drishti_depth_upgrade.md`  
**Scope:** Indoor-first navigation. Outdoor mode = manual toggle. All additions fit within confirmed server headroom (~16ms available, additions cost ~8–10ms).  
**Confirmed baseline:** Server inference ~102ms · Round-trip 160–200ms · FPS 5.0–6.2 · P0 budget 400ms

---

## Confirmed Runtime Constants (use these everywhere)

```python
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
LETTERBOX_SIZE    = 518          # Depth Anything V2 native input
YOLO_SIZE         = 640
JPEG_QUALITY      = 60
EMA_ALPHA         = 0.50         # Higher than outdoor (0.4) — compensates for lower FPS
ROLLING_WINDOW_N  = 8            # frames (~1.45s at 5.5 FPS average)
CEIL_MASK_FRAC    = 0.25         # top 25% of frame = ceiling, excluded from all analysis
GROUND_ROI_FRAC   = 0.25         # bottom 25% of frame = ground plane scan
ALERT_SUPPRESS_S  = 1.5          # indoor suppression window (seconds), shorter than outdoor 3s
YOLO_CONF_INDOOR  = 0.30         # lowered from 0.50
YOLO_TEMPORAL_MIN = 2            # detection must appear in N consecutive frames before alert fires
```

---

## Mode Toggle

### Android side — `MainActivity.kt`

Add a toggle button to the UI. On tap, send a mode message over the existing WebSocket:

```kotlin
// Add to UI
var indoorMode = true  // default

fun onModeToggle() {
    indoorMode = !indoorMode
    val msg = JSONObject()
    msg.put("type", "mode")
    msg.put("mode", if (indoorMode) "indoor" else "outdoor")
    webSocket.send(msg.toString())
    tts.speak(if (indoorMode) "Indoor mode" else "Outdoor mode", QUEUE_FLUSH, null, null)
}
```

### Server side — `main.py`

```python
# Global state
current_mode = "indoor"  # default

# In WebSocket message handler, before frame processing
if msg.get("type") == "mode":
    current_mode = msg["mode"]
    return  # no frame to process

# Pass mode into process_frame
result = process_frame(frame_bytes, mode=current_mode)
```

All pipeline logic below gates on `mode == "indoor"`. Outdoor mode calls the existing pipeline unchanged.

---

## Step 1 — Ceiling Mask

**File:** `depth_utils.py`  
**Cost:** <1ms  
**Run:** Before any depth map analysis, every frame.

```python
def apply_ceiling_mask(depth_map: np.ndarray, ceil_frac: float = CEIL_MASK_FRAC) -> np.ndarray:
    """
    Zero out the top ceil_frac of the depth map.
    Ceiling pixels are textureless and produce unstable depth values indoors.
    Returns a copy with ceiling region set to NaN (excluded from all stats).
    """
    masked = depth_map.copy().astype(float)
    cutoff = int(depth_map.shape[0] * ceil_frac)
    masked[:cutoff, :] = np.nan
    return masked
```

Apply immediately after depth map is produced, before rolling window, before grid, before everything else.

---

## Step 2 — Rolling Window Normalisation (indoor-tuned)

**File:** `depth_utils.py`  
**Cost:** ~1ms  
**Change from previous plan:** Window size = 8 frames (was 10). EMA alpha = 0.50 (was 0.40).

```python
from collections import deque
import numpy as np

depth_history = deque(maxlen=ROLLING_WINDOW_N)
window_ready  = False

def update_rolling_window(depth_map: np.ndarray):
    """Append masked depth map to history. Ignores NaN (ceiling) pixels."""
    depth_history.append(depth_map)

def get_normalised_depth(depth_map: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Returns (normalised_depth_map, window_ready).
    Normalises against rolling 5th/95th percentile of history.
    Falls back to per-frame normalisation during cold start.
    """
    global window_ready
    valid_pixels = depth_map[~np.isnan(depth_map)]

    if len(depth_history) < ROLLING_WINDOW_N:
        # Cold start — per-frame fallback
        scene_min = np.nanpercentile(depth_map, 5)
        scene_max = np.nanpercentile(depth_map, 95)
        window_ready = False
    else:
        all_vals = np.concatenate([d[~np.isnan(d)] for d in depth_history])
        scene_min = np.percentile(all_vals, 5)
        scene_max = np.percentile(all_vals, 95)
        window_ready = True

    denom = scene_max - scene_min
    if denom < 1e-6:
        denom = 1e-6

    normed = (depth_map - scene_min) / denom
    normed = np.clip(normed, 0.0, 1.0)
    normed[np.isnan(depth_map)] = np.nan  # preserve ceiling mask
    return normed, window_ready
```

**Cold-start audio cue:** If `window_ready == False`, include `"warming_up": true` in the JSON response. Android plays "Drishti warming up" once on first receive of this flag, then stays silent until `warming_up` is absent.

---

## Step 3 — EMA Temporal Smoothing

**File:** `depth_utils.py`  
**Cost:** <1ms

```python
ema_depth: np.ndarray | None = None

def apply_ema(depth_map: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """
    Exponential moving average across frames.
    alpha=0.5 at ~5.5 FPS ≈ 0.4 at 8 FPS in effective smoothing.
    NaN pixels (ceiling) are excluded — EMA only on valid pixels.
    """
    global ema_depth
    if ema_depth is None or ema_depth.shape != depth_map.shape:
        ema_depth = depth_map.copy()
        return ema_depth

    valid = ~np.isnan(depth_map)
    ema_depth[valid] = alpha * depth_map[valid] + (1 - alpha) * ema_depth[valid]
    ema_depth[~valid] = np.nan
    return ema_depth
```

Apply EMA **after** ceiling mask, **before** normalisation. Order: mask → EMA → normalise → grid.

---

## Step 4 — 3×2 Occupancy Grid

**File:** `indoor_grid.py`  
**Cost:** <1ms  
This is the core indoor perception primitive. Everything else reads from this grid.

```python
import numpy as np

# Grid layout (after ceiling mask applied):
# Row 0 = upper half of valid frame (walls, obstacles at head/torso height)
# Row 1 = lower half of valid frame (floor-level obstacles, approaching ground)
# Col 0 = left third, Col 1 = center third, Col 2 = right third

def build_occupancy_grid(normed_depth: np.ndarray) -> np.ndarray:
    """
    Returns a 2×3 array of median depth scores.
    NaN cells (ceiling) are excluded from median computation.
    Lower score = closer = more occupied.
    """
    h, w = normed_depth.shape
    ceil_cutoff = int(h * CEIL_MASK_FRAC)
    valid_h = h - ceil_cutoff
    mid_row  = ceil_cutoff + valid_h // 2

    row_splits = [ceil_cutoff, mid_row, h]
    col_splits = [0, w // 3, 2 * w // 3, w]

    grid = np.full((2, 3), np.nan)
    for r in range(2):
        for c in range(3):
            cell = normed_depth[row_splits[r]:row_splits[r+1],
                                col_splits[c]:col_splits[c+1]]
            valid = cell[~np.isnan(cell)]
            if valid.size > 0:
                grid[r, c] = np.median(valid)
    return grid

# Indoor depth tier thresholds (recalibrate from your corridor walk — these are starting points)
P0_THRESH = 0.25   # Stop
P1_THRESH = 0.50   # Slow
P2_THRESH = 0.70   # Caution

def grid_cell_tier(score: float) -> int:
    """Returns alert tier (0=P0, 1=P1, 2=P2, 3=clear) for a grid cell score."""
    if np.isnan(score):  return 3
    if score < P0_THRESH: return 0
    if score < P1_THRESH: return 1
    if score < P2_THRESH: return 2
    return 3
```

**Mandatory calibration step before using these thresholds:**
Walk toward a wall. Log `grid[0, 1]` (upper center cell) at 3m, 2m, 1m, 0.5m. Set P0_THRESH to the score observed at ~1m. Set P1_THRESH to score at ~2m. P2_THRESH at ~3m. Do the same for side columns (wall approach from side). Write the calibrated values into `config.py`, not hardcoded here.

---

## Step 5 — Corridor-Mode Zone Logic

**File:** `indoor_zones.py`  
**Cost:** <1ms  
Replaces vanishing-point zone logic entirely for indoor mode.

```python
import numpy as np
from indoor_grid import build_occupancy_grid, grid_cell_tier, P1_THRESH

NARROW_CORRIDOR_THRESH = P1_THRESH   # both walls P1 or closer = narrow corridor
NARROW_CORRIDOR_FRAMES = 5           # must persist this many frames to confirm narrow corridor

_narrow_counter = 0

def analyse_corridor(grid: np.ndarray) -> dict:
    """
    Analyses the 2×3 grid and returns zone guidance.

    Returns dict:
        mode:        "narrow_corridor" | "wall_left" | "wall_right" | "open_room"
        center_tier: int (tier of center column — the walking path)
        wall_left:   float (left column upper-half median score)
        wall_right:  float (right column upper-half median score)
        narrowing:   bool (center column score dropping = path closing in)
    """
    global _narrow_counter

    left_score   = grid[0, 0]   # upper-left  = left wall
    center_score = grid[0, 1]   # upper-center = path ahead
    right_score  = grid[0, 2]   # upper-right = right wall

    left_close  = (not np.isnan(left_score))  and left_score  < NARROW_CORRIDOR_THRESH
    right_close = (not np.isnan(right_score)) and right_score < NARROW_CORRIDOR_THRESH

    if left_close and right_close:
        _narrow_counter += 1
    else:
        _narrow_counter = max(0, _narrow_counter - 1)

    narrow = _narrow_counter >= NARROW_CORRIDOR_FRAMES

    return {
        "mode":         "narrow_corridor" if narrow else
                        ("wall_left"  if left_close  else
                        ("wall_right" if right_close else "open_room")),
        "center_tier":  grid_cell_tier(center_score),
        "wall_left":    float(left_score)  if not np.isnan(left_score)  else 1.0,
        "wall_right":   float(right_score) if not np.isnan(right_score) else 1.0,
        "narrowing":    False,  # populated by trend detector below
    }

# --- Trend detector: is the center column closing in? ---
_center_history = []
CENTER_TREND_WINDOW = 5   # frames

def update_center_trend(center_score: float) -> bool:
    """Returns True if center column depth score is consistently dropping (path closing in)."""
    _center_history.append(center_score)
    if len(_center_history) > CENTER_TREND_WINDOW:
        _center_history.pop(0)
    if len(_center_history) < CENTER_TREND_WINDOW:
        return False
    # Monotonically decreasing = closing in
    return all(_center_history[i] > _center_history[i+1]
               for i in range(len(_center_history) - 1))
```

**Narrow corridor alert logic:**
- In narrow corridor mode, suppress wall alerts entirely — both walls being close is expected.
- Only alert if `center_tier <= P1` OR `narrowing == True`.
- Alert message: "Path narrowing ahead" if narrowing, "Obstacle ahead" if center_tier P0.

---

## Step 6 — Ground-Plane Discontinuity Scan (Steps + Stairs)

**File:** `ground_scan.py`  
**Cost:** 2–3ms

```python
import numpy as np
from collections import deque

GROUND_GRAD_THRESH  = 0.18   # gradient magnitude spike threshold — calibrate from real step
GROUND_CLUSTER_MIN  = 15     # minimum pixels in a discontinuity cluster to fire
STAIR_FRAME_COUNT   = 3      # consecutive frames with same-band discontinuity = stairs

_discontinuity_history = deque(maxlen=STAIR_FRAME_COUNT)

def scan_ground_plane(normed_depth: np.ndarray,
                      yolo_boxes: list[dict]) -> dict | None:
    """
    Scans the bottom GROUND_ROI_FRAC of the frame for depth discontinuities.
    Excludes pixels inside YOLO bounding boxes (to avoid object edges).

    Returns dict with keys: type ("step"|"stairs"), zone ("left"|"center"|"right"), tier (int)
    Returns None if no discontinuity found.
    """
    h, w = normed_depth.shape
    roi_start = int(h * (1 - GROUND_ROI_FRAC))
    ground = normed_depth[roi_start:, :].copy()

    # Mask out YOLO box regions in ground ROI
    for box in yolo_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        # Remap to ground ROI coordinates
        y1_roi = max(0, y1 - roi_start)
        y2_roi = max(0, y2 - roi_start)
        if y2_roi > 0:
            ground[y1_roi:y2_roi, x1:x2] = np.nan

    # Compute gradient magnitude
    gy, gx = np.gradient(np.nan_to_num(ground, nan=0.0))
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Threshold
    spike_mask = grad_mag > GROUND_GRAD_THRESH
    spike_count = int(np.sum(spike_mask))

    if spike_count < GROUND_CLUSTER_MIN:
        _discontinuity_history.append(None)
        return None

    # Find dominant horizontal band of the discontinuity
    spike_rows, spike_cols = np.where(spike_mask)
    median_col = int(np.median(spike_cols))
    zone = "left" if median_col < w // 3 else ("right" if median_col > 2 * w // 3 else "center")

    _discontinuity_history.append(zone)

    # Stairs: same zone fires in all recent frames
    if (len(_discontinuity_history) == STAIR_FRAME_COUNT and
            all(z == zone for z in _discontinuity_history)):
        hazard_type = "stairs"
        tier = 0   # P0 — always stop for stairs
    else:
        hazard_type = "step"
        tier = 1   # P1

    return {"type": hazard_type, "zone": zone, "tier": tier}
```

**Calibration required:** Walk toward a known step. Log `grad_mag` peak values at 1m, 0.5m. Set `GROUND_GRAD_THRESH` to 80% of the peak at 1m. For stairs, walk toward a staircase and verify `_discontinuity_history` fills with the same zone for 3 consecutive frames.

---

## Step 7 — Doorway Detection

**File:** `indoor_zones.py` (add to existing file)  
**Cost:** 1–2ms

```python
DOORWAY_FAR_THRESH  = 0.65   # depth score — a "far" region in otherwise close upper frame
DOORWAY_MIN_WIDTH   = 60     # minimum pixel width of the gap (at 640px wide = ~10% of frame)

def detect_doorway(normed_depth: np.ndarray, corridor_info: dict) -> dict | None:
    """
    Looks for a vertical strip of far depth in the upper half of the frame.
    Only meaningful when corridor_info["mode"] in ("narrow_corridor", "wall_left", "wall_right").
    Returns dict with zone, or None.
    """
    if corridor_info["mode"] == "open_room":
        return None

    h, w = normed_depth.shape
    ceil_cutoff = int(h * CEIL_MASK_FRAC)
    upper_half  = normed_depth[ceil_cutoff: ceil_cutoff + (h - ceil_cutoff) // 2, :]

    # Column-wise median of upper half
    col_medians = np.nanmedian(upper_half, axis=0)

    # Find columns that are "far" (potential doorway gap)
    far_cols = np.where(col_medians > DOORWAY_FAR_THRESH)[0]

    if len(far_cols) < DOORWAY_MIN_WIDTH:
        return None

    # Find the widest contiguous run of far columns
    gaps = np.split(far_cols, np.where(np.diff(far_cols) > 5)[0] + 1)
    widest = max(gaps, key=len)

    center_col = int(np.mean(widest))
    zone = "left" if center_col < w // 3 else ("right" if center_col > 2 * w // 3 else "center")

    return {"zone": zone, "width_px": len(widest)}
```

**Alert message:** "Opening [zone]" — not "obstacle". Doorway alert is informational, not a hazard. Give it tier 3 (clear) so it never overrides a hazard alert, but always gets spoken if no higher-priority alert is active.

---

## Step 8 — Reflective Surface Detection (Glass / Mirror)

**File:** `indoor_zones.py`  
**Cost:** <1ms

```python
MIRROR_DEPTH_MIN    = 0.75   # depth model thinks this region is very far
MIRROR_YOLO_CONF    = 0.30   # YOLO fired on this region at any confidence

def check_mirror_anomaly(yolo_boxes: list[dict],
                         grid: np.ndarray,
                         corridor_info: dict) -> bool:
    """
    Returns True if a YOLO detection is in a region the depth map says is far —
    inside a corridor where that distance is physically impossible.

    Heuristic: if corridor width (from grid) implies max ~5m depth, but a YOLO
    box center maps to a grid cell with score > MIRROR_DEPTH_MIN, flag it.
    """
    # Only relevant in corridor modes
    if corridor_info["mode"] == "open_room":
        return False

    w = 640
    col_boundaries = [0, w // 3, 2 * w // 3, w]

    for box in yolo_boxes:
        box_center_x = (box["x1"] + box["x2"]) // 2
        col = 0 if box_center_x < col_boundaries[1] else \
              (2 if box_center_x > col_boundaries[2] else 1)
        cell_score = grid[0, col]
        if not np.isnan(cell_score) and cell_score > MIRROR_DEPTH_MIN:
            return True   # YOLO sees object where depth says nothing is there
    return False
```

If `check_mirror_anomaly` returns True: suppress the YOLO detection, include `"surface_warning": "reflective"` in the JSON response. Android plays "Reflective surface detected" once, then suppresses for 5 seconds.

---

## Step 9 — YOLO Temporal Consistency Filter

**File:** `yolo_utils.py`  
**Cost:** <1ms

```python
from collections import defaultdict, deque

_detection_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=YOLO_TEMPORAL_MIN))

def filter_detections_temporal(detections: list[dict]) -> list[dict]:
    """
    Only passes a detection if the same class has been detected in the same
    broad zone for YOLO_TEMPORAL_MIN consecutive frames.
    Kills one-frame noise without delaying real persistent objects.
    """
    w = 640
    current_keys = set()

    for det in detections:
        cx = (det["x1"] + det["x2"]) // 2
        zone = "L" if cx < w // 3 else ("R" if cx > 2 * w // 3 else "C")
        key = f"{det['class']}_{zone}"
        _detection_history[key].append(True)
        current_keys.add(key)

    # Age out keys not seen this frame
    for key in list(_detection_history.keys()):
        if key not in current_keys:
            _detection_history[key].append(False)

    # Pass only detections with full consecutive history
    confirmed = []
    for det in detections:
        cx = (det["x1"] + det["x2"]) // 2
        zone = "L" if cx < w // 3 else ("R" if cx > 2 * w // 3 else "C")
        key = f"{det['class']}_{zone}"
        if all(_detection_history[key]):
            confirmed.append(det)

    return confirmed
```

---

## Step 10 — Alert Composition (Indoor Priority Queue)

**File:** `alert_composer.py`  
**Cost:** <1ms

Priority order (highest to lowest). First match wins:

```python
from time import monotonic

_last_alert_time  = 0.0
_last_alert_text  = ""

PRIORITY = ["stairs", "wall_closing", "step", "obstacle", "wall", "doorway", "surface_warning"]

def compose_indoor_alert(
    corridor_info:  dict,
    ground_hazard:  dict | None,
    doorway:        dict | None,
    yolo_confirmed: list[dict],
    grid:           np.ndarray,
    mirror_anomaly: bool,
    window_ready:   bool,
) -> dict:
    """
    Returns the single highest-priority alert as a JSON-serialisable dict.
    Enforces ALERT_SUPPRESS_S suppression window between identical alerts.
    """
    global _last_alert_time, _last_alert_text
    now = monotonic()

    alerts = []

    # 1. Stairs (P0 always)
    if ground_hazard and ground_hazard["type"] == "stairs":
        alerts.append({"priority": 0, "text": f"Stairs ahead, stop",
                        "tier": 0, "pan": "center"})

    # 2. Wall closing in (center column trend + narrow corridor)
    if corridor_info["narrowing"] or (
            corridor_info["mode"] == "narrow_corridor" and corridor_info["center_tier"] <= 1):
        alerts.append({"priority": 1, "text": "Path narrowing",
                        "tier": corridor_info["center_tier"], "pan": "center"})

    # 3. Step
    if ground_hazard and ground_hazard["type"] == "step":
        alerts.append({"priority": 2,
                        "text": f"Step ahead {ground_hazard['zone']}",
                        "tier": ground_hazard["tier"],
                        "pan": ground_hazard["zone"]})

    # 4. YOLO obstacle (confirmed, non-mirror)
    if not mirror_anomaly:
        for det in yolo_confirmed:
            cx    = (det["x1"] + det["x2"]) // 2
            zone  = "left" if cx < FRAME_WIDTH // 3 else \
                    ("right" if cx > 2 * FRAME_WIDTH // 3 else "center")
            col   = 0 if zone == "left" else (2 if zone == "right" else 1)
            tier  = grid_cell_tier(grid[0, col])
            alerts.append({"priority": 3,
                            "text": f"{det['class']} {zone}",
                            "tier": tier, "pan": zone})

    # 5. Wall alert (single wall, not narrow corridor)
    if corridor_info["mode"] == "wall_left":
        alerts.append({"priority": 4, "text": "Wall on left",
                        "tier": 1, "pan": "left"})
    elif corridor_info["mode"] == "wall_right":
        alerts.append({"priority": 4, "text": "Wall on right",
                        "tier": 1, "pan": "right"})

    # 6. Doorway (informational)
    if doorway:
        alerts.append({"priority": 5,
                        "text": f"Opening {doorway['zone']}",
                        "tier": 3, "pan": doorway["zone"]})

    # 7. Reflective surface
    if mirror_anomaly:
        alerts.append({"priority": 6, "text": "Reflective surface",
                        "tier": 2, "pan": "center"})

    if not alerts:
        return {"alert": False, "warming_up": not window_ready}

    best = min(alerts, key=lambda a: a["priority"])

    # Suppression: same text within window → skip
    if best["text"] == _last_alert_text and (now - _last_alert_time) < ALERT_SUPPRESS_S:
        return {"alert": False, "warming_up": not window_ready}

    _last_alert_time = now
    _last_alert_text = best["text"]

    return {
        "alert":       True,
        "text":        best["text"],
        "tier":        best["tier"],
        "pan":         best["pan"],
        "warming_up":  not window_ready,
    }
```

---

## Step 11 — Main Pipeline Integration

**File:** `main.py` — replace `process_frame()` body for indoor mode

```python
import numpy as np
from depth_utils   import apply_ceiling_mask, apply_ema, update_rolling_window, get_normalised_depth
from indoor_grid   import build_occupancy_grid, grid_cell_tier
from indoor_zones  import analyse_corridor, update_center_trend, detect_doorway, check_mirror_anomaly
from ground_scan   import scan_ground_plane
from yolo_utils    import filter_detections_temporal
from alert_composer import compose_indoor_alert

def process_frame_indoor(frame_bgr: np.ndarray,
                         raw_yolo_detections: list[dict],
                         raw_depth_map: np.ndarray) -> dict:
    """
    Full indoor pipeline. Called after YOLO and Depth Anything have already run in parallel.
    All steps below are pure NumPy — total added cost ~8–10ms.
    """

    # 1. Ceiling mask
    masked_depth = apply_ceiling_mask(raw_depth_map)

    # 2. EMA smoothing
    smoothed_depth = apply_ema(masked_depth)

    # 3. Rolling window normalisation
    update_rolling_window(smoothed_depth)
    normed_depth, window_ready = get_normalised_depth(smoothed_depth)

    # 4. 3×2 occupancy grid
    grid = build_occupancy_grid(normed_depth)

    # 5. Corridor zone logic
    corridor_info = analyse_corridor(grid)
    corridor_info["narrowing"] = update_center_trend(grid[0, 1])

    # 6. Ground-plane scan
    ground_hazard = scan_ground_plane(normed_depth, raw_yolo_detections)

    # 7. Doorway detection
    doorway = detect_doorway(normed_depth, corridor_info)

    # 8. Mirror / reflective surface check
    mirror_anomaly = check_mirror_anomaly(raw_yolo_detections, grid, corridor_info)

    # 9. YOLO temporal filter
    yolo_confirmed = filter_detections_temporal(raw_yolo_detections)

    # 10. Alert composition
    alert = compose_indoor_alert(
        corridor_info  = corridor_info,
        ground_hazard  = ground_hazard,
        doorway        = doorway,
        yolo_confirmed = yolo_confirmed,
        grid           = grid,
        mirror_anomaly = mirror_anomaly,
        window_ready   = window_ready,
    )

    return alert
```

---

## Debug Overlay (development only)

Send this extra block in the JSON response when `DEBUG_MODE = True` in `config.py`. Android renders it as a coloured 3×2 grid overlay on the camera preview. Remove before demo.

```python
def build_debug_payload(grid: np.ndarray, corridor_info: dict) -> dict:
    tier_colours = {0: "red", 1: "orange", 2: "yellow", 3: "green"}
    cells = []
    for r in range(2):
        for c in range(3):
            cells.append({
                "row": r, "col": c,
                "score": round(float(grid[r, c]), 3) if not np.isnan(grid[r, c]) else None,
                "tier":  grid_cell_tier(grid[r, c]),
                "colour": tier_colours[grid_cell_tier(grid[r, c])],
            })
    return {"grid": cells, "corridor_mode": corridor_info["mode"]}
```

---

## Calibration Protocol (run before any threshold tuning)

**Duration:** ~15 minutes in your test corridor.  
**Log:** Save raw `grid[0,1]` values and `grad_mag` peaks to a CSV during each walk.

| Walk | What to measure | Threshold to set |
|---|---|---|
| Toward wall, straight | `grid[0,1]` at 3m / 2m / 1m / 0.5m | `P0_THRESH`, `P1_THRESH`, `P2_THRESH` |
| Side-step toward left wall | `grid[0,0]` at 1m, 0.5m | `NARROW_CORRIDOR_THRESH` |
| Toward a single step | `grad_mag` peak at 1m, 0.5m | `GROUND_GRAD_THRESH` |
| Toward a doorway | `col_medians` gap width at 2m, 1m | `DOORWAY_MIN_WIDTH` |
| Point at mirror / glass | `grid` scores vs YOLO boxes | `MIRROR_DEPTH_MIN` |

Write all calibrated values into `config.py`. Never hardcode thresholds in logic files.

---

## Build Order

Execute strictly in this sequence. Each step is independently testable with the debug overlay before proceeding.

```
1. apply_ceiling_mask         → verify ceiling excluded in debug overlay
2. apply_ema + rolling window → verify grid stops flickering
3. build_occupancy_grid       → run calibration walk, set thresholds in config.py
4. analyse_corridor           → test: walk toward wall, check wall_left/wall_right fires
5. update_center_trend        → test: walk down corridor, check narrowing flag appears
6. scan_ground_plane          → test: walk toward a step, check step/stairs distinction
7. detect_doorway             → test: walk toward open door, check "Opening center" fires
8. check_mirror_anomaly       → test: point at glass door, check reflective surface flag
9. filter_detections_temporal → test: wave hand briefly, check no alert fires
10. compose_indoor_alert      → full integration test: walk full corridor end-to-end
11. mode toggle               → test: toggle indoor→outdoor→indoor, verify mode switch audio
```

---

## What Is Not Changed

- YOLO model weights — unchanged
- Depth Anything V2 model — unchanged  
- Letterbox + coordinate remap — carry over from `drishti_depth_upgrade.md` unchanged
- Binaural pan logic — unchanged, pan channel driven by zone from `alert_composer`
- Sherpa-ONNX TTS — unchanged
- WebSocket protocol — unchanged
- Android CameraX pipeline — unchanged except mode toggle button
