"""
Drishti Server Configuration
All tuneable parameters in one place.
"""

# ── Network ──────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── Model paths / IDs ────────────────────────────────────────────────────────
YOLO_MODEL_PATH = "yolov8n.pt"  # auto-downloads on first run
# Depth model checkpoints are in checkpoints/ — loaded by models.py

# ── Input resolutions ────────────────────────────────────────────────────────
YOLO_INPUT_SIZE = (640, 640)
# Depth model handles its own 518×518 resize internally — no config needed

# ── Depth thresholds (METRIC — absolute meters) ─────────────────────────────
# Used by resolve_tier_metric() for YOLO detections
METRIC_P0_M = 1.0    # < 1.0m  → STOP
METRIC_P1_M = 2.0    # < 2.0m  → Slow down
METRIC_P2_M = 4.0    # < 4.0m  → Warning
METRIC_P3_M = 6.0    # < 6.0m  → Guidance
# > 6.0m → suppress alert

# ── YOLO confidence threshold ────────────────────────────────────────────────
YOLO_CONFIDENCE = 0.40

# ── Path-clear watchdog ──────────────────────────────────────────────────────
PATH_CLEAR_TIMEOUT_SEC = 5.0  # seconds of no P0/P1 before sending "Path is clear"

# ── Debug / Calibration ──────────────────────────────────────────────────────
CALIBRATION_MODE = False  # set True to print depth/class/bbox per detection (dev only)
DEBUG_MODE = False        # if True, sends debug grid overlay in JSON response

# ── Indoor pipeline ──────────────────────────────────────────────────────────
CEIL_MASK_FRAC    = 0.25    # top 25% of frame = ceiling, excluded from analysis
GROUND_ROI_FRAC   = 0.25    # bottom 25% of frame = ground plane scan
ALERT_SUPPRESS_S  = 1.5     # indoor suppression window (seconds)
YOLO_CONF_INDOOR  = 0.30    # lowered from 0.40 for indoor
YOLO_TEMPORAL_MIN = 2       # detection must appear in N consecutive frames

# ── Indoor alert thresholds (METRIC — absolute metres) ────────────────────────
P0_THRESH_M = 0.65   # adjusted for ±0.2m sensor error margin
P1_THRESH_M = 1.0
P2_THRESH_M = 2.0
P3_THRESH_M = 3.5

# ── Looming detector ──────────────────────────────────────────────────────────
LOOMING_DEPTH_M    = 0.6    # raw depth trigger for looming override
LOOMING_RATE_MS    = 0.5    # m/s approach rate required for looming override
LOOMING_HISTORY_N  = 3      # frames for rolling median (deque maxlen)

# ── P3 suppression ────────────────────────────────────────────────────────────
P3_SUPPRESS_S = 10.0   # same P3 message suppressed for 10 seconds
