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
DEBUG_MODE = False        # if True, sends debug grid overlay in JSON response

# ── Indoor pipeline ──────────────────────────────────────────────────────────
CEIL_MASK_FRAC    = 0.25    # top 25% of frame = ceiling, excluded from analysis
GROUND_ROI_FRAC   = 0.25    # bottom 25% of frame = ground plane scan
EMA_ALPHA         = 0.50    # higher than outdoor — compensates for lower FPS
ROLLING_WINDOW_N  = 8       # frames (~1.45s at 5.5 FPS average)
ALERT_SUPPRESS_S  = 1.5     # indoor suppression window (seconds)
YOLO_CONF_INDOOR  = 0.30    # lowered from 0.40 for indoor
YOLO_TEMPORAL_MIN = 2       # detection must appear in N consecutive frames

# ── Indoor grid thresholds (calibrate from corridor walk) ────────────────────
P0_THRESH = 0.15   # Stop — object extremely close (< arm's reach)
P1_THRESH = 0.35   # Slow — object approaching (1-2m)
P2_THRESH = 0.60   # Caution — object at medium distance

