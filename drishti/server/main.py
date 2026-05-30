"""
Drishti FastAPI Server — main entry point.

Endpoints:
    GET  /health  — server status + CUDA availability
    WS   /ws      — main processing pipeline (JPEG in, JSON out)

Run:
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import io
import time
import json
import logging

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
)
from models import load_models, run_yolo, run_depth, get_device
from depth_processing import get_depth_score, resolve_tier, find_generic_obstacles
from zone_detection import get_zone, get_pan_channel
from alert_builder import build_message, select_highest_priority

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

    last_p0_p1_time = time.time()
    path_clear_sent = False
    frame_count = 0

    try:
        while True:
            # ── 1. Receive JPEG frame ────────────────────────────────────────
            data = await ws.receive_bytes()
            t_start = time.perf_counter()
            frame_count += 1

            # ── 2. Decode ────────────────────────────────────────────────────
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # ── 3. Resize for each model ─────────────────────────────────────
            yolo_img = img.resize(YOLO_INPUT_SIZE)
            depth_img = img.resize(DEPTH_INPUT_SIZE)

            # ── 4. Run YOLO ──────────────────────────────────────────────────
            t_yolo = time.perf_counter()
            detections = run_yolo(yolo_img)
            t_yolo_done = time.perf_counter()

            # ── 5. Run Depth ─────────────────────────────────────────────────
            t_depth = time.perf_counter()
            depth_map = run_depth(depth_img)
            t_depth_done = time.perf_counter()

            # ── 6. Process detections ────────────────────────────────────────
            alerts = []
            depth_h, depth_w = depth_map.shape[:2]

            for det in detections:
                label = det["label"]
                # Scale bbox from YOLO input space to depth map space
                bx1, by1, bx2, by2 = det["bbox"]
                sx = depth_w / YOLO_INPUT_SIZE[0]
                sy = depth_h / YOLO_INPUT_SIZE[1]
                scaled_bbox = (
                    int(bx1 * sx),
                    int(by1 * sy),
                    int(bx2 * sx),
                    int(by2 * sy),
                )

                depth_score = get_depth_score(depth_map, scaled_bbox)
                tier = resolve_tier(depth_score, label)

                if tier is None:
                    continue

                zone = get_zone(scaled_bbox, depth_w)
                pan = get_pan_channel(zone)
                
                # Only objects directly in the CENTER can trigger a P0 (Immediate Stop).
                # Side objects are capped at P1 (Slow down / Warning).
                if tier == "P0" and zone != "CENTER":
                    tier = "P1"
                    
                message = build_message(tier, label, zone)

                alerts.append({
                    "tier": tier,
                    "label": label,
                    "message": message,
                    "pan_channel": pan,
                    "depth_score": depth_score,
                    "zone": zone,
                })

                if CALIBRATION_MODE:
                    bbox_area = (bx2 - bx1) * (by2 - by1)
                    logger.info(
                        f"[CAL] depth_score={depth_score:.3f} "
                        f"class={label} tier={tier} zone={zone} "
                        f"bbox_area={bbox_area}"
                    )

            # ── 6.5 Generic Obstacle Detection ───────────────────────────────
            generic_obstacles = find_generic_obstacles(depth_map)
            for obs in generic_obstacles:
                obs_zone = obs["zone"]
                obs_depth = obs["depth_score"]
                
                # Check if YOLO already found something in this zone that is similarly close or closer
                # We add a 0.1 buffer so YOLO takes precedence even if its box depth is slightly further
                collision = any(
                    a["zone"] == obs_zone and a["depth_score"] <= obs_depth + 0.15
                    for a in alerts
                )
                
                if not collision:
                    pan = get_pan_channel(obs_zone)
                    message = build_message(obs["tier"], obs["label"], obs_zone)
                    alerts.append({
                        "tier": obs["tier"],
                        "label": obs["label"],
                        "message": message,
                        "pan_channel": pan,
                        "depth_score": obs_depth,
                        "zone": obs_zone,
                    })
                    
                    if CALIBRATION_MODE:
                        logger.info(
                            f"[CAL] GENERIC OBSTACLE depth_score={obs_depth:.3f} "
                            f"tier={obs['tier']} zone={obs_zone}"
                        )

            # ── 7. Select highest priority ───────────────────────────────────
            best = select_highest_priority(alerts)

            if best:
                has_high = best["tier"] in ("P0", "P1")
                if has_high:
                    last_p0_p1_time = time.time()
                    path_clear_sent = False

                payload = {
                    "tier": best["tier"],
                    "message": best["message"],
                    "pan_channel": best["pan_channel"],
                }
                await ws.send_text(json.dumps(payload))

            else:
                # ── 8. Path-clear watchdog ────────────────────────────────────
                elapsed = time.time() - last_p0_p1_time
                if elapsed >= PATH_CLEAR_TIMEOUT_SEC and not path_clear_sent:
                    payload = {
                        "tier": "P4",
                        "message": "Path is clear",
                        "pan_channel": 0.0,
                    }
                    await ws.send_text(json.dumps(payload))
                    path_clear_sent = True
                    last_p0_p1_time = time.time()
                else:
                    # No alert, no path-clear — send empty ack so client
                    # can open the frame gate for the next frame
                    await ws.send_text(json.dumps({"tier": None, "message": None, "pan_channel": 0.0}))

            # ── Timing log ───────────────────────────────────────────────────
            t_total = (time.perf_counter() - t_start) * 1000
            if frame_count % 30 == 0 or CALIBRATION_MODE:
                logger.info(
                    f"Frame {frame_count}: "
                    f"YOLO {(t_yolo_done - t_yolo) * 1000:.1f}ms | "
                    f"Depth {(t_depth_done - t_depth) * 1000:.1f}ms | "
                    f"Total {t_total:.1f}ms | "
                    f"Detections: {len(detections)} | "
                    f"Alert: {best['tier'] + ' ' + best['label'] if best else 'none'}"
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"Session ended. Processed {frame_count} frames.")


# ── Run server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
