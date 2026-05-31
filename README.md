# Drishti — AI Navigation Assistant for the Visually Impaired

> **Real-time obstacle detection and spatial audio guidance, streamed from a Python server to an Android device.**

Drishti uses a camera-equipped Android phone to stream frames to a local Python inference server, which runs YOLO object detection and metric depth estimation in parallel, then sends back prioritised audio alerts over WebSocket.

---

## Architecture

```
Android (Kotlin)  ──JPEG frames──►  FastAPI Server (Python)
                  ◄──JSON alerts──        │
                                     ┌────┴────────────┐
                                     │  YOLO v8n       │  Object detection
                                     │  Depth-Anything │  Metric depth (meters)
                                     │  V2 (Metric)    │
                                     └────┬────────────┘
                                          │
                              ┌───────────┼──────────────┐
                           Indoor       Outdoor        Ground
                           pipeline     pipeline       scan
```

### Pipelines

| Mode | What runs |
|------|-----------|
| **Outdoor** | Vanishing point → Cross-validator → Ground-plane scan → Generic obstacle fallback |
| **Indoor**  | Ceiling mask → 3×2 occupancy grid → Corridor/Doorway/Mirror analysis → YOLO temporal filter |

All depth values are **absolute meters** (no normalisation needed).

---

## Quick Start

### Server (Python 3.10+)

```bash
cd drishti/server
pip install -r requirements.txt
python main.py
# Server listens on ws://0.0.0.0:8000/ws
```

> **GPU recommended.** The server runs on CPU but inference will be slow (~1–2 FPS). With a CUDA GPU expect 10–15 FPS.

### Android App

1. Open `drishti/android/` in Android Studio (Hedgehog or newer).
2. Set the server IP in `MainActivity.kt` (search for `WS_URL`).
3. Build and run on a device running Android 8+ (API 26+).

> **Model checkpoints:** Download `depth_anything_v2_metric_hypersim_vits.pth`
> and `depth_anything_v2_metric_vkitti_vits.pth` from HuggingFace and place
> them in `drishti/server/checkpoints/`.

> **Sherpa-ONNX TTS** is optional. If you want on-device TTS instead of Android's built-in `TextToSpeech`, see `drishti/android/app/src/main/assets/README.md` for asset setup instructions.

---

## Alert Tiers

| Tier | Meaning | Distance |
|------|---------|----------|
| P0 | **Stop now** | < 1 m |
| P1 | **Slow down** | 1 – 2 m |
| P2 | **Caution** | 2 – 4 m |
| P3 | **Guidance / info** | 4 – 6 m |
| P4 | System messages | — |

---

## Server Configuration

All tuneable parameters live in [`drishti/server/config.py`](drishti/server/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PORT` | `8000` | WebSocket port |
| `METRIC_P0_M` | `1.0` | Stop threshold (metres) |
| `METRIC_P1_M` | `2.0` | Slow threshold |
| `YOLO_CONFIDENCE` | `0.40` | Outdoor detection confidence |
| `CALIBRATION_MODE` | `False` | Set `True` to log depth/class/tier per detection |
| `PATH_CLEAR_TIMEOUT_SEC` | `5.0` | Seconds before "Path is clear" is announced |

---

## Project Structure

```
drishti/
├── server/
│   ├── main.py              # FastAPI server, WebSocket pipeline
│   ├── config.py            # All tuneable parameters
│   ├── depth_processing.py  # Metric depth → tier resolution + generic obstacle fallback
│   ├── alert_builder.py     # Priority selection across mixed alert types
│   ├── alert_composer.py    # Indoor alert priority queue
│   ├── ground_scan.py       # Outdoor kerb/pothole/drain detector
│   ├── indoor_zones.py      # Corridor/doorway/mirror analysis
│   ├── class_tiers.py       # COCO class → danger tier mapping
│   └── requirements.txt
└── android/
    └── app/src/main/
        ├── java/com/drishti/   # Kotlin source
        └── assets/             # Sherpa-ONNX TTS model files (optional)
```

---

## Requirements

### Server
- Python 3.10+
- PyTorch ≥ 2.0 (CUDA optional)
- See `drishti/server/requirements.txt`

### Android
- Android Studio Hedgehog (2023.1.1)+
- Android API 26+ (Android 8.0)
- JDK 17+ (bundled with Android Studio)

---

## Hackathon Notes

- The Android build requires **JDK 17** (not JDK 26). Use the JDK bundled with Android Studio via `File → Project Structure → SDK Location → Gradle JDK`.
- Sherpa-ONNX AAR and TTS model assets are **not committed** to the repo (large binary files). Follow `assets/README.md` to download them.
- The server auto-downloads `yolov8n.pt` on first run if not present.
- Connect the Android device to the **laptop's personal hotspot** (not venue Wi-Fi).
  The server IP on a Windows hotspot is always `192.168.137.1`.
