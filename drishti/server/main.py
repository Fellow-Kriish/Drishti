"""
Drishti FastAPI Server — main entry point.

Endpoints:
    GET  /health  — server status + CUDA availability
    WS   /ws      — main processing pipeline (JPEG in, JSON out)

Pipeline (v0.3 — Indoor + Outdoor):
    Shared: Letterbox → YOLO + Depth (parallel)
    Indoor: Ceiling mask → EMA → Rolling norm → Grid → Corridor/Doorway/Mirror → Alert
    Outdoor: Rolling norm → VP estimation → Cross-validation → Ground scan → Alert

Run:
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from config import (
    HOST,
    PORT,
    YOLO_INPUT_SIZE,
    DEPTH_INPUT_SIZE,
    CALIBRATION_MODE,
    PATH_CLEAR_TIMEOUT_SEC,
    YOLO_CONF_INDOOR,
    YOLO_CONFIDENCE,
)
from models import load_models, run_yolo, run_depth, get_device
from depth_processing import resolve_tier
from zone_detection import get_zone, get_pan_channel
from alert_builder import build_message, select_highest_priority
from class_tiers import TIER_ORDER

# ── Outdoor modules (Depth Upgrade v0.2) ────────────────────────────────────
from utils.letterbox import letterbox, remap_bbox
from depth_normalizer import DepthNormalizer
from cross_validator import validated_tier
from open_path import open_path_direction
from ground_scan import scan_ground_plane
from vanishing_point import VanishingPointEstimator

# ── Indoor modules (v0.3) ───────────────────────────────────────────────────
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
    """Load models on startup."""
    load_models()
    logger.info(f"Server ready on {HOST}:{PORT}")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Drishti Server", lifespan=lifespan)


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": get_device(),
    }


# ── WebSocket processing pipeline ───────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected.")

    # ── Session state ────────────────────────────────────────────────────────
    current_mode = "indoor"  # default mode
    last_p0_p1_time = time.time()
    path_clear_sent = False
    frame_count = 0

    # Outdoor singletons
    normalizer   = DepthNormalizer()
    vp_estimator = VanishingPointEstimator(frame_width=DEPTH_INPUT_SIZE[0])

    # Indoor singletons
    indoor_depth     = IndoorDepthProcessor()
    indoor_zones     = IndoorZoneAnalyser()
    indoor_ground    = IndoorGroundScanner()
    yolo_filter      = YoloTemporalFilter()
    alert_composer   = IndoorAlertComposer()

    try:
        while True:
            # ── Receive message ──────────────────────────────────────────────
            message = await ws.receive()

            # Handle text messages (mode toggle)
            if "text" in message:
                try:
                    msg_data = json.loads(message["text"])
                    if msg_data.get("type") == "mode":
                        current_mode = msg_data.get("mode", "indoor")
                        logger.info(f"Mode switched to: {current_mode}")
                        await ws.send_text(json.dumps({
                            "tier": "P4",
                            "message": f"{current_mode.capitalize()} mode",
                            "pan_channel": 0.0,
                        }))
                except (json.JSONDecodeError, KeyError):
                    pass
                continue

            # Handle binary messages (JPEG frames)
            if "bytes" not in message:
                continue

            data = message["bytes"]
            t_start = time.perf_counter()
            frame_count += 1

            # ── Decode ───────────────────────────────────────────────────────
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # ── Letterbox resize ─────────────────────────────────────────────
            yolo_img,  yolo_meta  = letterbox(img, YOLO_INPUT_SIZE[0])
            depth_img, depth_meta = letterbox(img, DEPTH_INPUT_SIZE[0])

            yolo_pil  = Image.fromarray(yolo_img)
            depth_pil = Image.fromarray(depth_img)

            # ── Run YOLO + Depth in parallel ─────────────────────────────────
            t_inference = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                yolo_future  = executor.submit(run_yolo, yolo_pil)
                depth_future = executor.submit(run_depth, depth_pil)
                detections_raw = yolo_future.result()
                depth_map      = depth_future.result()
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
                    depth_map, detections_raw, yolo_meta, depth_meta,
                    normalizer, vp_estimator,
                    last_p0_p1_time, path_clear_sent,
                    frame_count,
                )
                # Update watchdog state from outdoor result
                if payload.get("_has_high"):
                    last_p0_p1_time = time.time()
                    path_clear_sent = False
                elif payload.get("_path_clear_sent"):
                    path_clear_sent = True
                    last_p0_p1_time = time.time()

                # Remove internal keys before sending
                payload.pop("_has_high", None)
                payload.pop("_path_clear_sent", None)

            await ws.send_text(json.dumps(payload))

            # ── Timing log ───────────────────────────────────────────────────
            t_total = (time.perf_counter() - t_start) * 1000
            t_infer = (t_inference_done - t_inference) * 1000
            if frame_count % 30 == 0 or CALIBRATION_MODE:
                alert_str = payload.get("message") or "none"
                tier_str = payload.get("tier") or "-"
                logger.info(
                    f"Frame {frame_count} [{current_mode}]: "
                    f"Inference {t_infer:.1f}ms | "
                    f"Total {t_total:.1f}ms | "
                    f"Dets: {len(detections_raw)} | "
                    f"Payload: {len(data)}B | "
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
    """Full indoor pipeline — all steps are pure NumPy, ~8-10ms total."""

    # 1-3. Ceiling mask → EMA → Rolling normalization
    normed_depth, window_ready = indoor_depth.process(depth_map)

    if not window_ready:
        if frame_count % 5 == 0:
            logger.info(f"Frame {frame_count}: Indoor warming up")
        return {
            "tier": "P4",
            "message": "Drishti warming up",
            "pan_channel": 0.0,
        }

    # 4. 3×2 occupancy grid
    grid = build_occupancy_grid(normed_depth)

    # 5. Corridor zone logic
    corridor_info = indoor_zones.analyse_corridor(grid)
    corridor_info["narrowing"] = indoor_zones.update_center_trend(grid[0, 1])

    # 6. Ground-plane scan (steps / stairs)
    ground_hazard = indoor_ground.scan(normed_depth, detections_raw)

    # 7. Doorway detection
    doorway = detect_doorway(normed_depth, corridor_info)

    # 8. Mirror / reflective surface check
    mirror_anomaly = check_mirror_anomaly(detections_raw, grid, corridor_info)

    # 9. YOLO temporal filter
    yolo_confirmed = yolo_filter.filter(detections_raw)

    # 10. Alert composition
    alert = alert_composer.compose(
        corridor_info=corridor_info,
        ground_hazard=ground_hazard,
        doorway=doorway,
        yolo_confirmed=yolo_confirmed,
        grid=grid,
        mirror_anomaly=mirror_anomaly,
        window_ready=window_ready,
    )

    if CALIBRATION_MODE:
        logger.info(
            f"[CAL-INDOOR] mode={corridor_info['mode']} "
            f"center_tier={corridor_info['center_tier']} "
            f"narrowing={corridor_info['narrowing']} "
            f"ground={ground_hazard['type'] if ground_hazard else 'none'} "
            f"doorway={doorway['zone'] if doorway else 'none'} "
            f"mirror={mirror_anomaly} "
            f"yolo_confirmed={len(yolo_confirmed)}"
        )

    # Format response
    if alert.get("alert"):
        return {
            "tier": alert["tier"],
            "message": alert["message"],
            "pan_channel": alert["pan_channel"],
        }
    else:
        return {
            "tier": None,
            "message": None,
            "pan_channel": 0.0,
        }


# ── Outdoor pipeline ────────────────────────────────────────────────────────

def _process_outdoor(
    depth_map: np.ndarray,
    detections_raw: list[dict],
    yolo_meta: dict,
    depth_meta: dict,
    normalizer: DepthNormalizer,
    vp_estimator: VanishingPointEstimator,
    last_p0_p1_time: float,
    path_clear_sent: bool,
    frame_count: int,
) -> dict:
    """Existing outdoor pipeline from v0.2, extracted into a function."""

    # Rolling depth normalization
    normalizer.update(depth_map)

    if not normalizer.ready:
        if frame_count % 5 == 0:
            logger.info(
                f"Frame {frame_count}: Outdoor warming up "
                f"({len(normalizer._mins)}/{normalizer.WINDOW_FRAMES} frames)"
            )
        return {
            "tier": "P4",
            "message": "Drishti warming up",
            "pan_channel": 0.0,
        }

    depth_norm = normalizer.normalize(depth_map)

    # Vanishing point estimation
    vp_x = vp_estimator.update(depth_norm)
    col_bounds = vp_estimator.column_boundaries(vp_x)

    # Process detections
    alerts = []
    depth_h, depth_w = depth_map.shape[:2]
    processed_detections = []

    for det in detections_raw:
        label = det["label"]
        bbox_yolo = det["bbox"]

        bbox_depth = remap_bbox(bbox_yolo, yolo_meta, depth_meta)
        depth_score = normalizer.score(depth_norm, bbox_depth)
        tier = resolve_tier(depth_score, label)

        if tier is None:
            processed_detections.append({"bbox_depth": bbox_depth, "bbox_yolo": bbox_yolo})
            continue

        final_tier = validated_tier(depth_score, bbox_yolo, label, tier)
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
            "depth_score": depth_score,
            "zone": zone,
        })
        processed_detections.append({"bbox_depth": bbox_depth, "bbox_yolo": bbox_yolo})

        if CALIBRATION_MODE:
            bbox_area = (bbox_yolo[2] - bbox_yolo[0]) * (bbox_yolo[3] - bbox_yolo[1])
            logger.info(
                f"[CAL] depth_score={depth_score:.3f} "
                f"class={label} raw_tier={tier} final_tier={final_tier} "
                f"zone={zone} bbox_area={bbox_area}"
            )

    # Ground-plane scan
    ground_alert = scan_ground_plane(depth_norm, processed_detections, vanishing_point_x=vp_x)
    if ground_alert:
        alerts.append(ground_alert)

    # Open-path direction
    path_msg = open_path_direction(depth_norm, processed_detections, col_boundaries=col_bounds)

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
            return {
                "tier": None,
                "message": None,
                "pan_channel": 0.0,
            }


# ── Run server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
