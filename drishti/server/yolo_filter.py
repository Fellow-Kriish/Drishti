"""
YOLO temporal consistency filter for indoor mode.

Only passes a detection if the same class has been detected in the same
broad zone for YOLO_TEMPORAL_MIN consecutive frames.
Kills one-frame noise without delaying real persistent objects.
"""

from collections import defaultdict, deque

from config import YOLO_TEMPORAL_MIN


class YoloTemporalFilter:
    """Session-scoped YOLO detection filter with frame history."""

    def __init__(self):
        self._history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=YOLO_TEMPORAL_MIN)
        )

    def filter(self, detections: list[dict]) -> list[dict]:
        """
        Filter detections requiring temporal consistency.

        Args:
            detections: list of dicts with 'label' and 'bbox' keys.

        Returns:
            Only detections that appeared in the same zone for
            YOLO_TEMPORAL_MIN consecutive frames.
        """
        w = 640
        current_keys = set()

        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            zone = "L" if cx < w // 3 else ("R" if cx > 2 * w // 3 else "C")
            key = f"{det['label']}_{zone}"
            self._history[key].append(True)
            current_keys.add(key)

        # Age out keys not seen this frame
        for key in list(self._history.keys()):
            if key not in current_keys:
                self._history[key].append(False)

        # Pass only detections with full consecutive history
        confirmed = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            zone = "L" if cx < w // 3 else ("R" if cx > 2 * w // 3 else "C")
            key = f"{det['label']}_{zone}"
            if len(self._history[key]) >= YOLO_TEMPORAL_MIN and all(self._history[key]):
                confirmed.append(det)

        return confirmed
