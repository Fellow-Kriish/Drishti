"""
Drishti FastAPI Server — main entry point.

Endpoints:
    GET  /health  — server status + CUDA availability
    WS   /ws      — main processing pipeline (JPEG in, JSON out)

Pipeline (v0.4 — Metric Depth):
    Shared: cv2 decode → YOLO (letterboxed PIL) + Depth (BGR raw) in parallel
    Indoor: Ceiling mask → Grid → Corridor/Doorway/Mirror → Alert
    Outdoor: VP estimation → Cross-validation → Ground scan → Alert

    All depth values are absolute meters. No warmup phase needed.

Run:
    python main.py
"""

import io
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from config import (
    HOST,
    PORT,
    YOLO_INPUT_SIZE,
    CALIBRATION_MODE,
    PATH_CLEAR_TIMEOUT_SEC,
)
from models import load_models, run_yolo, run_depth, get_device, swap_depth_model
from depth_processing import resolve_tier_metric, get_depth_meters
from zone_detection import get_zone, get_pan_channel
from alert_builder import build_message, select_highest_priority
from class_tiers import TIER_ORDER

# ── Outdoor modules ─────────────────────────────────────────────────────────
from utils.letterbox import letterbox, remap_bbox
from cross_validator import validated_tier
from open_path import open_path_direction
from ground_scan import scan_ground_plane
from vanishing_point import VanishingPointEstimator

# ── Indoor modules ──────────────────────────────────────────────────────────
from depth_utils import IndoorDepthProcessor
from indoor_grid import build_occupancy_grid
from indoor_zones import IndoorZoneAnalyser, detect_doorway, check_mirror_anomaly
from indoor_ground_scan import IndoorGroundScanner
from yolo_filter import YoloTemporalFilter
from alert_composer import IndoorAlertComposer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("drishti.server")


# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    logger.info(f"Server ready on {HOST}:{PORT}")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Drishti Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "device": get_device()}


# ── Helper: decode JPEG to BGR numpy array ───────────────────────────────────

def decode_frame(jpeg_bytes: bytes) -> np.ndarray:
    """Returns BGR numpy array. Do NOT convert to RGB — metric model expects BGR."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("imdecode failed — malformed JPEG")
    return frame


# ── WebSocket processing pipeline ───────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected.")

    # ── Session state ────────────────────────────────────────────────────────
    current_mode = "indoor"
    last_p0_p1_time = time.time()
    path_clear_sent = False
    frame_count = 0

    # Outdoor
    vp_estimator = VanishingPointEstimator(frame_width=518)

    # Indoor
    indoor_depth     = IndoorDepthProcessor()
    indoor_zones     = IndoorZoneAnalyser()
    indoor_ground    = IndoorGroundScanner()
    yolo_filter      = YoloTemporalFilter()
    alert_composer   = IndoorAlertComposer()

    try:
        while True:
            message = await ws.receive()

            # Handle text messages (mode toggle)
            if "text" in message:
                try:
                    msg_data = json.loads(message["text"])
                    if msg_data.get("type") == "mode":
                        current_mode = msg_data.get("mode", "indoor")
                        swap_depth_model(current_mode)
                        logger.info(f"Mode switched to: {current_mode}")
                        await ws.send_text(json.dumps({
                            "tier": "P4",
                            "message": f"{current_mode.capitalize()} mode",
                            "pan_channel": 0.0,
                        }))
                except (json.JSONDecodeError, KeyError):
                    pass
                continue

            if "bytes" not in message:
                continue

            data = message["bytes"]
            t_start = time.perf_counter()
            frame_count += 1

            # ── Decode to BGR ────────────────────────────────────────────────
            frame_bgr = decode_frame(data)

            # ── Prepare YOLO input (letterboxed PIL) ─────────────────────────
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            yolo_img, yolo_meta = letterbox(pil_img, YOLO_INPUT_SIZE[0])
            yolo_pil = Image.fromarray(yolo_img)

            # ── Run YOLO + Depth in parallel ─────────────────────────────────
            t_inference = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future  = executor.submit(run_yolo, yolo_pil)
                depth_future = executor.submit(run_depth, frame_bgr)
                detections_raw = yolo_future.result()
                depth_map      = depth_future.result()  # HxW in METERS
            t_inference_done = time.perf_counter()

            # ── Route to indoor or outdoor pipeline ──────────────────────────
            if current_mode == "indoor":
                payload = _process_indoor(
                    depth_map, detections_raw,
                    indoor_depth, indoor_zones, indoor_ground,
                    yolo_filter, alert_composer,
                    frame_count,
                )
            else:
                payload = _process_outdoor(
                    depth_map, detections_raw, yolo_meta, frame_bgr.shape,
                    vp_estimator,
                    last_p0_p1_time, path_clear_sent,
                    frame_count,
                )
                if payload.get("_has_high"):
                    last_p0_p1_time = time.time()
                    path_clear_sent = False
                elif payload.get("_path_clear_sent"):
                    path_clear_sent = True
                    last_p0_p1_time = time.time()
                payload.pop("_has_high", None)
                payload.pop("_path_clear_sent", None)

            await ws.send_text(json.dumps(payload))

            # ── Timing log ───────────────────────────────────────────────────
            t_total = (time.perf_counter() - t_start) * 1000
            t_infer = (t_inference_done - t_inference) * 1000
            if frame_count % 30 == 0 or CALIBRATION_MODE:
                alert_str = payload.get("message") or "none"
                tier_str = payload.get("tier") or "-"
                depth_range = f"{depth_map.min():.1f}-{depth_map.max():.1f}m"
                logger.info(
                    f"Frame {frame_count} [{current_mode}]: "
                    f"Inference {t_infer:.1f}ms | "
                    f"Total {t_total:.1f}ms | "
                    f"Dets: {len(detections_raw)} | "
                    f"Depth: {depth_range} | "
                    f"Alert: {tier_str} {alert_str}"
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"Session ended. Processed {frame_count} frames.")


# ── Indoor pipeline ─────────────────────────────────────────────────────────

def _process_indoor(
    depth_map: np.ndarray,
    detections_raw: list[dict],
    indoor_depth: IndoorDepthProcessor,
    indoor_zones: IndoorZoneAnalyser,
    indoor_ground: IndoorGroundScanner,
    yolo_filter: YoloTemporalFilter,
    alert_composer: IndoorAlertComposer,
    frame_count: int,
) -> dict:
    """Full indoor pipeline — metric depth, no warmup needed."""

    # 1. Ceiling mask (still useful — ceilings produce noisy depth)
    masked_depth = indoor_depth.process(depth_map)

    # 2. 3×2 occupancy grid (values in meters now)
    grid = build_occupancy_grid(masked_depth)

    # 3. Corridor zone logic
    corridor_info = indoor_zones.analyse_corridor(grid)
    corridor_info["narrowing"] = indoor_zones.update_center_trend(grid[0, 1])

    # 4. Ground-plane scan (steps / stairs)
    ground_hazard = indoor_ground.scan(masked_depth, detections_raw)

    # 5. Doorway detection
    doorway = detect_doorway(masked_depth, corridor_info)

    # 6. Mirror / reflective surface check
    mirror_anomaly = check_mirror_anomaly(detections_raw, grid, corridor_info)

    # 7. YOLO temporal filter
    yolo_confirmed = yolo_filter.filter(detections_raw)

    # 8. Alert composition
    alert = alert_composer.compose(
        corridor_info=corridor_info,
        ground_hazard=ground_hazard,
        doorway=doorway,
        yolo_confirmed=yolo_confirmed,
        grid=grid,
        mirror_anomaly=mirror_anomaly,
        window_ready=True,  # always ready with metric depth
    )

    if CALIBRATION_MODE:
        logger.info(
            f"[CAL-INDOOR] mode={corridor_info['mode']} "
            f"center={grid[0,1]:.1f}m tier={corridor_info['center_tier']} "
            f"narrowing={corridor_info['narrowing']} "
            f"ground={ground_hazard['type'] if ground_hazard else 'none'} "
            f"doorway={doorway['zone'] if doorway else 'none'} "
            f"mirror={mirror_anomaly} "
            f"yolo={len(yolo_confirmed)}"
        )

    if alert.get("alert"):
        return {
            "tier": alert["tier"],
            "message": alert["message"],
            "pan_channel": alert["pan_channel"],
        }
    else:
        return {"tier": None, "message": None, "pan_channel": 0.0}


# ── Outdoor pipeline ────────────────────────────────────────────────────────

def _process_outdoor(
    depth_map: np.ndarray,
    detections_raw: list[dict],
    yolo_meta: dict,
    frame_shape: tuple,
    vp_estimator: VanishingPointEstimator,
    last_p0_p1_time: float,
    path_clear_sent: bool,
    frame_count: int,
) -> dict:
    """Outdoor pipeline — metric depth, no normalizer needed."""

    depth_h, depth_w = depth_map.shape[:2]

    # Vanishing point estimation (works on raw depth geometry)
    vp_x = vp_estimator.update(depth_map)
    col_bounds = vp_estimator.column_boundaries(vp_x)

    # Build depth_meta for bbox remapping (depth map is same size as input frame)
    depth_meta = {
        "scale": 1.0,
        "pad_x": 0,
        "pad_y": 0,
        "orig_w": depth_w,
        "orig_h": depth_h,
    }

    alerts = []
    processed_detections = []

    for det in detections_raw:
        label = det["label"]
        bbox_yolo = det["bbox"]

        bbox_depth = remap_bbox(bbox_yolo, yolo_meta, depth_meta)
        depth_m = get_depth_meters(depth_map, bbox_depth)
        tier = resolve_tier_metric(depth_m, label)

        if tier is None:
            processed_detections.append({"bbox_depth": bbox_depth, "bbox_yolo": bbox_yolo})
            continue

        final_tier = validated_tier(depth_m, bbox_yolo, label, tier)
        zone = get_zone(bbox_depth, depth_w, col_boundaries=col_bounds)
        pan = get_pan_channel(zone)

        if final_tier == "P0" and zone != "CENTER":
            final_tier = "P1"

        message = build_message(final_tier, label, zone)
        alerts.append({
            "tier": final_tier,
            "label": label,
            "message": message,
            "pan_channel": pan,
            "depth_m": depth_m,
            "zone": zone,
        })
        processed_detections.append({"bbox_depth": bbox_depth, "bbox_yolo": bbox_yolo})

        if CALIBRATION_MODE:
            bbox_area = (bbox_yolo[2] - bbox_yolo[0]) * (bbox_yolo[3] - bbox_yolo[1])
            logger.info(
                f"[CAL] depth={depth_m:.2f}m "
                f"class={label} tier={final_tier} "
                f"zone={zone} bbox_area={bbox_area}"
            )

    # Ground-plane scan
    ground_alert = scan_ground_plane(depth_map, processed_detections, vanishing_point_x=vp_x)
    if ground_alert:
        alerts.append(ground_alert)

    # Open-path direction
    path_msg = open_path_direction(depth_map, processed_detections, col_boundaries=col_bounds)

    # Select highest priority
    best = select_highest_priority(alerts)

    if best:
        has_high = best["tier"] in ("P0", "P1")
        return {
            "tier": best["tier"],
            "message": best["message"],
            "pan_channel": best["pan_channel"],
            "_has_high": has_high,
        }
    else:
        elapsed = time.time() - last_p0_p1_time
        if elapsed >= PATH_CLEAR_TIMEOUT_SEC and not path_clear_sent:
            return {
                "tier": "P3",
                "message": path_msg,
                "pan_channel": 0.0,
                "_path_clear_sent": True,
            }
        else:
            return {"tier": None, "message": None, "pan_channel": 0.0}


# ── Run server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
