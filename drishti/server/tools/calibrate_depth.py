"""
Depth score calibration tool.

Walk toward a wall at known distances and log depth scores.
Use the CSV output to set tier thresholds in resolve_tier.

Usage:
    Import log_calibration and call it per detection during a calibration walk.
    Then pull the CSV and set P0/P1/P2/P3 score boundaries from real data.
"""

import csv
import time

LOG_FILE = "calibration_log.csv"


def log_calibration(depth_score: float, label: str, zone: str = "", tier: str = ""):
    """Log a single calibration measurement."""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            f"{depth_score:.4f}",
            label,
            zone,
            tier,
            "MEASURE_MANUALLY",
        ])
