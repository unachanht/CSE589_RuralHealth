# edge_server.py
import uvicorn
import threading
import time
import json
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from adaptive_stream import PriorityFrame, BandwidthEstimator, JPEGCompressor, AdaptiveScheduler
from edge_orchestrator import EdgeModelRunner

# =====================================================
# GLOBAL EDGE SERVER STATE
# =====================================================
app = FastAPI()

# These will be initialized in main()
EDGE_RUNNER = None
SCHEDULER = None
FRAME_BUFFER = []
FRAME_ID = 0
CLOUD_URL = None
STOP_THREADS = False

# Background transmission thread
def transmission_worker():
    global SCHEDULER, STOP_THREADS, CLOUD_URL

    print("[edge-server] Transmission worker started")

    while not STOP_THREADS:
        rec = SCHEDULER.transmit_once()

        if rec is None:
            time.sleep(0.05)
            continue

        print(f"[edge-server] tx frame={rec['frame_id']} lvl={rec['level']} "
              f"tier={rec['tier']} bytes={rec['bytes']} estbps={rec['est_bps_after']:.0f}")

        # -------------------------
        # Upload to cloud
        # -------------------------
        if CLOUD_URL:
            import requests
            files = {"frame": ("frame.jpg", rec["payload_bytes"], rec["mime"])}
            metadata = json.dumps({k:v for k,v in rec.items() if k!="payload_bytes"})

            try:
                r = requests.post(CLOUD_URL, files=files, data={"meta": metadata}, timeout=5)
                r.raise_for_status()
                response = r.json()

                # Cloud requests refinement
                for req in response.get("next_requests", []):
                    fid = req["frame_id"]
                    print(f"[edge-server] Cloud requests refinement for frame {fid}")

                    # Re-enqueue the frame with boosted priority
                    for fr in FRAME_BUFFER:
                        if fr.frame_id == fid:
                            boosted = PriorityFrame(
                                priority=fr.priority + 15.0,
                                frame_id=fr.frame_id,
                                timestamp=time.time(),
                                frame=fr.frame,
                                roi_mask=fr.roi_mask,
                                max_level=fr.max_level
                            )
                            SCHEDULER.enqueue_frames([boosted])
                            break

            except Exception as e:
                print("[edge-server] Cloud upload failed:", e)

        time.sleep(0.01)

    print("[edge-server] Transmission worker stopped")


# =====================================================
# API: RECEIVE FRAMES
# =====================================================
@app.post("/submit_frame")
async def submit_frame(frame: UploadFile):
    global FRAME_ID, FRAME_BUFFER, EDGE_RUNNER, SCHEDULER

    arr = np.frombuffer(await frame.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"status": "error", "msg": "invalid image"}, status_code=400)

    # 1. Edge inference
    score, roi = EDGE_RUNNER.infer(img)

    # 2. Wrap into PriorityFrame
    pf = PriorityFrame(
        priority=score,
        frame_id=FRAME_ID,
        timestamp=time.time(),
        frame=img,
        roi_mask=roi,
        max_level=2
    )
    FRAME_ID += 1

    FRAME_BUFFER.append(pf)
    SCHEDULER.enqueue_frames([pf])

    return {
        "status": "ok",
        "frame_id": pf.frame_id,
        "score": pf.priority
    }


@app.get("/status")
def status():
    return {
        "queued_frames": len(SCHEDULER.heap),
        "est_bps": SCHEDULER.estimator.get()
    }


# =====================================================
# ENTRY POINT
# =====================================================
def start_edge_server(
    model_path="edge_telemed_int8.pt",
    device="cpu",
    cloud_url="http://localhost:9000/upload_frame",
    initial_mbps=1.0
):
    global EDGE_RUNNER, SCHEDULER, CLOUD_URL, STOP_THREADS

    EDGE_RUNNER = EdgeModelRunner(model_path=model_path, device=device)
    estimator = BandwidthEstimator(initial_bps=initial_mbps * 1e6)
    compressor = JPEGCompressor(codec="jpeg")
    SCHEDULER = AdaptiveScheduler(estimator, compressor)
    CLOUD_URL = cloud_url

    STOP_THREADS = False
    worker = threading.Thread(target=transmission_worker, daemon=True)
    worker.start()

    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    start_edge_server()
