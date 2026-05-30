"""
Drishti Server Configuration
All tuneable parameters in one place.
"""

# ── Network ──────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── Model paths / IDs ────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"  # auto-downloads on first run
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# ── Input resolutions ────────────────────────────────────────────────────────
YOLO_INPUT_SIZE = (640, 640)
DEPTH_INPUT_SIZE = (518, 518)

# ── Depth thresholds (calibrate empirically) ─────────────────────────────────
# depth_score: 0.0 = closest, 1.0 = farthest
DEPTH_VERY_CLOSE = 0.15   # override to P0 regardless of class
DEPTH_CLOSE = 0.30        # escalate one tier
DEPTH_MEDIUM = 0.55       # use base tier as-is
# > 0.55 → suppress alert (too far)

# ── YOLO confidence threshold ────────────────────────────────────────────────
YOLO_CONFIDENCE = 0.40

# ── Path-clear watchdog ──────────────────────────────────────────────────────
PATH_CLEAR_TIMEOUT_SEC = 5.0  # seconds of no P0/P1 before sending "Path is clear"

# ── Debug / Calibration ──────────────────────────────────────────────────────
CALIBRATION_MODE = True  # if True, prints depth_score/class/bbox per frame
