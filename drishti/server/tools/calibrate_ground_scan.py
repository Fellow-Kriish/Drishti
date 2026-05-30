"""
Ground scan calibration tool.

Walk toward known hazards (kerbs, drains) and flat ground.
Log gradient magnitudes to set GRADIENT_THRESHOLD.

Usage:
    Import log_ground_scan and call it during a calibration walk.
    Collect rows for label="flat" (noise floor), label="kerb", label="drain".
    Set GRADIENT_THRESHOLD = flat_max * 1.5 (headroom above noise).
"""

import csv
import time

import numpy as np

LOG = "ground_calibration.csv"


def log_ground_scan(grad_mag: np.ndarray, label: str):
    """Log gradient magnitude statistics for a ground region."""
    region_max  = float(grad_mag.max())
    region_mean = float(grad_mag.mean())
    with open(LOG, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            label,
            f"{region_max:.4f}",
            f"{region_mean:.4f}",
        ])
