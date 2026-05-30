# Drishti (दृष्टि) — AI Navigation Assistant for the Visually Impaired
**Team Antigravity | Final Development Plan**

---

## Problem

- 34M visually impaired Indians, 90% low-income
- Primary tool: white cane — no tech upgrade in decades
- Indian streets: open drains, vendors, animals, no tactile paving
- Western apps (Seeing AI, Lookout) built for structured roads — unusable here

---

## Hardware

| Component | Spec |
|---|---|
| Android device | Demo phone running Kotlin app |
| Laptop GPU | NVIDIA RTX 2050 4GB VRAM |
| Network | Laptop Wi-Fi hotspot (no venue Wi-Fi dependency) |

### GPU validation checklist (run before writing server code)
```python
import torch
print(torch.cuda.is_available())       # must be True
print(torch.cuda.get_device_name(0))   # must show RTX 2050
```
- Load both models, run 100 frames, print per-frame time
- Watch for thermal throttling after minute 3–4
- Do not run Chrome/Discord/OBS on the same machine during the demo
- Expected inference times: YOLOv8n ~6–10ms, Depth Anything V2 Small ~80–120ms
- Expected VRAM usage: ~2.5–3GB loaded simultaneously (1–1.5GB headroom)

---

## Network Setup (Option 3: Laptop Hotspot)

Skip venue Wi-Fi entirely. The laptop creates its own hotspot; the phone connects to it.

### Windows
- Settings → Mobile Hotspot → enable
- Laptop IP on its own hotspot is always `192.168.137.1`

### Ubuntu
```bash
nmcli dev wifi hotspot ifname wlan0 ssid Drishti password drishti123
# Laptop IP on hotspot interface is typically 10.42.0.1
```

### Android client
- Hardcode the hotspot IP in a `constants.kt` file (not scattered inline)
- Add a simple settings screen with an editable IP field as fallback
- Persist the IP with `SharedPreferences`

```kotlin
// constants.kt
object Config {
    var serverIp: String = "192.168.137.1"  // laptop hotspot IP
    var serverPort: Int = 8000
    val wsUrl get() = "ws://$serverIp:$serverPort/ws"
}
```

**Why this works:** The laptop's IP on its own hotspot is deterministic and never changes. No DHCP surprises, no firewall issues, no venue network dependency. The phone can still use mobile data for internet since it only routes ML traffic through the hotspot.

---

## Tech Stack

### Android Client
| Component | Technology |
|---|---|
| Language | Kotlin |
| Camera | CameraX |
| WebSocket | OkHttp |
| TTS Engine | Sherpa-ONNX (Kotlin API) |
| TTS Model | Piper ONNX — `vits-piper-en_US-libritts_r-medium` |
| Audio Playback | Android AudioTrack (raw PCM) |
| Assets | `tokens.txt`, `espeak-ng-data/` bundled in APK |

### Laptop Server
| Component | Technology |
|---|---|
| Language | Python |
| Framework | FastAPI |
| Object Detection | YOLOv8n |
| Depth Estimation | Depth Anything V2 Small |
| Deep Learning | PyTorch (CUDA) |
| Transport | WebSockets (JPEG in, JSON out) |

---

## Pipeline

```
Android Camera (5–10 FPS)
  → Resize to 640x480 JPEG @ 75% quality
  → WebSocket → FastAPI Server
      → Resize for YOLO: 640x640
      → Resize for Depth: 518x518
      → YOLOv8n: bounding boxes + labels
      → Depth Anything V2: depth map
      → 10th percentile depth inside bounding box
      → Normalize depth score against full frame
      → CLASS_TO_BASE_TIER lookup
      → Proximity escalation logic
      → Zone detection: LEFT / CENTER / RIGHT
      → Build JSON payload (highest priority object only)
  → WebSocket → Android
      → Parse JSON → Priority Queue
      → Sherpa-ONNX TTS → float[] → PCM bytes
      → AudioTrack with stereo pan
      → Vibration (P0 only)
      → Log to CSV
```

### Frame rate note
Target 8–10 FPS if inference budget allows. At 5 FPS, a bicycle at 1.5m gives ~1 second of reaction time — borderline for fast-moving obstacles. Benchmark on your RTX 2050 and push the rate as high as latency permits.

### JPEG quality note
Use 75% quality instead of 60%. The extra ~15KB per frame is negligible on a local hotspot. Depth Anything V2 is sensitive to high-frequency artifacts at object edges — exactly where depth boundaries matter — so the quality improvement is worth it.

---

## JSON Payload Schema

```json
{
  "tier": "P1",
  "message": "Slow, person ahead",
  "pan_channel": 0.0
}
```

| Field | Type | Values |
|---|---|---|
| `tier` | String | P0 / P1 / P2 / P3 / P4 |
| `message` | String | Human-readable alert |
| `pan_channel` | Float | -1.0 (Left), 0.0 (Center), 1.0 (Right) |

**Multi-object rule:** When multiple objects are detected in a single frame, send only the single highest-priority alert (highest tier, then closest depth score). This prevents audio queue flooding.

---

## YOLO Class → Tier Mapping

Define this table explicitly in `class_tiers.py` before writing any other server logic.

```python
# class_tiers.py
CLASS_TO_BASE_TIER = {
    # P0 — immediate stop
    "car": "P0",
    "truck": "P0",
    "bus": "P0",
    "motorcycle": "P0",
    "auto_rickshaw": "P0",   # fine-tune or alias from COCO if needed

    # P1 — slow down
    "person": "P1",
    "dog": "P1",
    "cow": "P1",
    "bicycle": "P1",

    # P2 — warning
    "cat": "P2",
    "chair": "P2",
    "bench": "P2",
    "potted plant": "P2",

    # P3 — guidance
    "traffic light": "P3",
    "stop sign": "P3",
}

# Objects not in this table → no alert
```

**Note on Indian-specific objects:** COCO includes motorcycle and bicycle but not auto-rickshaw. Either alias it from the closest COCO class or fine-tune YOLOv8n on a small Indian traffic dataset if time allows.

---

## Depth Thresholding (Relative → Proximity Tier)

Depth Anything V2 outputs relative disparity, not metric distance. Use percentile-based normalization to build a proximity classifier that works without calibration hardware.

```python
import numpy as np

def get_depth_score(depth_map, bbox):
    """
    Returns a normalized depth score 0.0–1.0.
    Lower = closer to camera.
    """
    x1, y1, x2, y2 = bbox
    region = depth_map[y1:y2, x1:x2]

    # 10th percentile catches the closest surface of the object.
    # Median misses thin/edge hazards like drain edges and poles.
    object_depth = np.percentile(region, 10)

    frame_min = depth_map.min()
    frame_max = depth_map.max()
    normalized = (object_depth - frame_min) / (frame_max - frame_min + 1e-6)

    return normalized  # lower = closer


def resolve_tier(depth_score, object_class):
    """
    Escalates or suppresses the base tier based on proximity.
    Thresholds below are starting points — calibrate empirically.
    """
    base_tier = CLASS_TO_BASE_TIER.get(object_class)
    if base_tier is None:
        return None  # unknown class, no alert

    TIER_ORDER = ["P0", "P1", "P2", "P3", "P4"]

    if depth_score < 0.15:
        # Very close — override to P0 regardless of class
        return "P0"
    elif depth_score < 0.30:
        # Close — escalate one tier
        idx = TIER_ORDER.index(base_tier)
        return TIER_ORDER[max(0, idx - 1)]
    elif depth_score < 0.55:
        # Medium distance — use base tier as-is
        return base_tier
    else:
        # Far — suppress alert
        return None
```

### Calibration session (do this once, ~20 minutes)
Walk toward a wall with the phone. Log depth scores at known distances.

```python
# Add temporarily to server for calibration
print(f"depth_score={depth_score:.3f} class={label} distance=MEASURE_THIS_MANUALLY")
```

Set your thresholds from that data. Lighting conditions affect relative depth — calibrate in the same environment you'll demo in.

---

## 5-Tier Audio Priority Queue

| Tier | Priority | Behavior | Example |
|---|---|---|---|
| P0 | Emergency | `AudioTrack.flush()` + instant play + vibrate | "Stop now, obstacle" |
| P1 | Critical | `AudioTrack.flush()` + instant play | "Slow, person ahead" |
| P2 | Warning | Enqueue — plays after current | "Bicycle on left" |
| P3 | Guidance | Enqueue | "Turn slightly right" |
| P4 | Ambient | Enqueue | "Path is clear" |

P2–P4 use `ConcurrentLinkedQueue` drained by a background coroutine. No overlapping voices.

### Deduplication logic (revised)

The 3-second blanket suppression is replaced with tier-aware deduplication:

- Suppress repeat alerts only if **the same object is in the same zone at the same or lower tier**
- Allow tier escalation (P2 → P1 → P0) to **always break suppression immediately**
- Reset suppression when the object's depth score changes by more than 0.10 (object is closing in)

```kotlin
data class AlertKey(val objectClass: String, val zone: String)

val suppressedAlerts = mutableMapOf<AlertKey, Pair<String, Long>>() // key → (tier, timestamp)

fun shouldSuppress(key: AlertKey, newTier: String): Boolean {
    val (lastTier, lastTime) = suppressedAlerts[key] ?: return false
    val elapsed = System.currentTimeMillis() - lastTime
    val isEscalating = TIER_ORDER.indexOf(newTier) < TIER_ORDER.indexOf(lastTier)
    return elapsed < 3000 && !isEscalating
}
```

---

## Spatial Audio (Binaural Panning)

Uses `AudioTrack.setStereoVolume(left, right)` before writing PCM buffer.

| `pan_channel` | Left Volume | Right Volume |
|---|---|---|
| -1.0 (Left) | 1.0 | 0.1 |
| 0.0 (Center) | 1.0 | 1.0 |
| 1.0 (Right) | 0.1 | 1.0 |

Piper voice physically sounds like it is coming from the obstacle's direction.

---

## Sherpa-ONNX Integration (Android)

1. **Dependency** — `implementation 'com.k2-fsa:sherpa-onnx:1.12.11'`
2. **Assets** — Place in `src/main/assets/`: Piper ONNX model, `tokens.txt`, `espeak-ng-data/`
3. **Init** — `OfflineTtsConfig` via `AssetManager` during app startup
4. **AudioTrack** — `ENCODING_PCM_16BIT`, sample rate = model rate (16000Hz or 22050Hz)
5. **Generate** — `OfflineTts.generate(text)` → float array → convert to PCM `ByteArray` → write to AudioTrack

---

## All Features

### Core
- Real-time obstacle detection via YOLOv8n
- Monocular depth estimation via Depth Anything V2
- Percentile-based proximity scoring (10th percentile, normalized per frame)
- Tier escalation based on proximity — not just object class
- Zone-aware spatial alerts (Left / Center / Right)
- Binaural panning — voice comes from obstacle direction
- Offline TTS via Sherpa-ONNX + Piper (no internet needed for speech)

### Safety & Reliability
- **Laptop hotspot networking** — deterministic IP, no venue Wi-Fi dependency
- **Server offline fallback** — if WebSocket drops and reconnection fails beyond max backoff, trigger a continuous gentle P0 beep so the user knows to stop, not just that connection is lost
- **WebSocket reconnection** — exponential backoff (1s → 2s → 4s → max 16s), P0 "Connection lost" alert on drop, P1 "Reconnected" on restore
- **In-flight frame gate** — `AtomicBoolean` prevents frame pileup; next frame only sent after JSON response received
- **Audio focus** — `AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK` — ducks Google Maps, calls, etc.
- **Vibration fallback** — P0 triggers 3-pulse pattern (200ms on / 100ms off × 3) in parallel with audio

### UX
- **Startup self-test** — "Drishti ready. Camera active." plays on init; WebSocket connects only after TTS confirmed
- **"Path is clear" watchdog** — P4 fires every 5 seconds of no P0/P1 detections; silence is never ambiguous
- **Alert deduplication** — tier-aware: same object + same zone suppressed for 3 seconds unless tier escalates; escalation always fires immediately
- **Settings screen** — editable server IP field, persisted via `SharedPreferences`; pre-filled with hotspot default

### Debug
- **CSV session log** — every alert logged as `timestamp, tier, message, pan_channel, depth_score`
- Saved to `/Android/data/com.yourapp/files/drishti_log_YYYYMMDD_HHmmss.csv`
- Pull via Android Studio Device Explorer or `adb pull`
- **Server-side calibration mode** — toggle in `config.py` to print `depth_score, class, bbox_area` per frame for threshold tuning

---

## Camera + Lifecycle Handling

- CameraX bound in `onResume()`, unbound in `onPause()`
- WebSocket reconnects in `onResume()`, closed in `onPause()`
- Never start CameraX in `onCreate()`
- Lock camera to a fixed orientation or read device orientation from sensor to remap zone boundaries — do not assume portrait = horizontal FOV

---

## Server-Side Frame Processing

```python
# Two models, two resize targets — done server-side
yolo_input  = img.resize((640, 640))   # YOLO native resolution
depth_input = img.resize((518, 518))   # Depth Anything V2 native resolution

# Run both models
yolo_results = yolo_model(yolo_input)
depth_map    = depth_model(depth_input)

# Per detection
alerts = []
for det in yolo_results:
    label = det.label
    bbox  = det.bbox  # scaled to depth_map resolution
    score = get_depth_score(depth_map, bbox)
    tier  = resolve_tier(score, label)
    zone  = get_zone(bbox, depth_map.width)
    if tier:
        alerts.append((tier, label, score, zone))

# Send only highest priority alert
if alerts:
    alerts.sort(key=lambda a: TIER_ORDER.index(a[0]))
    best = alerts[0]
    send_json(best)
```

---

## "Path Clear" Watchdog Logic (Server)

```
if no P0/P1 detections for >= 5 seconds:
    send { tier: "P4", message: "Path is clear", pan_channel: 0.0 }
    reset timer
else:
    send nothing this frame
```

---

## Latency Budget

Target: **< 400ms frame-to-audio for P0 alerts.**

| Stage | Target |
|---|---|
| Frame capture → sent | ~10ms |
| WebSocket transmission (hotspot) | ~5ms |
| JPEG decode + resize | ~5ms |
| YOLOv8n inference (RTX 2050) | ~10ms |
| Depth Anything V2 Small (RTX 2050) | ~100ms |
| Post-processing + JSON build | ~5ms |
| WebSocket response → Android | ~5ms |
| TTS generation (Sherpa-ONNX) | ~50ms |
| AudioTrack buffer → audio start | ~10ms |
| **Total** | **~200ms** |

Log timestamps at every stage during development. Add `depth_score` and `tier` to the CSV log so you can correlate latency with detection quality post-walk.

---

## Build Order

1. **Laptop server** — FastAPI + YOLO + Depth Anything, test with static image first; confirm CUDA is used
2. **Hotspot network** — set up laptop hotspot, confirm phone connects, confirm WebSocket connects to `192.168.137.1:8000`
3. **WebSocket pipeline** — connect Android camera, confirm frames flow and JSON returns; log latency at each stage
4. **Depth thresholding** — integrate `get_depth_score` and `resolve_tier`; run calibration session, set thresholds
5. **Class-to-tier table** — finalize `CLASS_TO_BASE_TIER`; test with real objects in your environment
6. **Sherpa-ONNX TTS** — get Piper speaking from hardcoded string on device
7. **Audio queue + panning** — wire JSON into queue, add stereo pan + vibration; implement revised deduplication
8. **Resilience layer** — reconnection, frame gate, audio focus, server-offline fallback beep
9. **Polish** — startup self-test, path-clear watchdog, CSV logger, settings screen

---

## Demo Strategy

- Use laptop hotspot — set it up before entering the venue, never touch venue Wi-Fi
- Control the environment — pre-set obstacles for clean P0/P1 triggers
- Show `AudioTrack.flush()` preemption live — have P0 interrupt a P2 mid-sentence
- Put a judge in headphones — obstacle on left → voice physically comes from left
- Have end-to-end latency number ready (measure frame sent → audio starts during prep)
- Show a real CSV log from a test walk
- Demo moment: "the caregiver sets the server IP once on a settings screen — after that the device is fully autonomous"

---

## Known Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| Depth is relative, not metric | Percentile normalization + empirical threshold calibration |
| Depth unreliable for thin objects (drains, poles) | 10th percentile instead of median captures closest surface |
| YOLO misses Indian-specific objects (auto-rickshaw) | Alias from closest COCO class; fine-tune if time allows |
| Lighting changes affect depth scores | Calibrate in demo environment; thresholds tuned per scene type |
| RTX 2050 may throttle under sustained load | Run 10-min stress test pre-hackathon; close all other apps |
| Fast obstacles (bicycle, motorcycle) at 5 FPS | Push to 8–10 FPS if latency budget allows |
| Camera orientation affects zone mapping | Lock orientation or read sensor to remap zone boundaries |
