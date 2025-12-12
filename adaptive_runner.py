# adaptive_runner.py
import os
import time
import json
import requests
from PIL import Image
import io

FRAMES_DIR = "frames"
CLOUD_URL = "http://127.0.0.1:8001/submit_frame"


# ---------------------------
# Tier configuration
# ---------------------------
def tier_to_params(tier):
    """
    Maps a tier to (priority, scale, quality)
    Matches expected fields for run_exp.py plots:
        - resolution vs bandwidth (scale)
        - jpeg quality vs bandwidth (quality)
        - transmission bytes vs priority
    """

    if tier == 1:
        return 1, 1.00, 95
    elif tier == 2:
        return 2, 0.75, 85
    elif tier == 3:
        return 3, 0.50, 70
    else:
        return 4, 0.40, 60


# ---------------------------
# Resize & compress image according to scale + quality
# ---------------------------
def prepare_image(frame_path, scale, quality):
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)))

    # compress to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------
# Cloud request wrapper
# ---------------------------
def send_to_cloud(frame_id, img_bytes, priority, scale, quality, tier):
    files = {
        "frame": ("frame.jpg", img_bytes, "image/jpeg")
    }

    meta = {
        "frame_id": frame_id,
        "priority": priority,
        "scale": scale,
        "quality": quality,
        "tier": tier,
        "bytes": len(img_bytes)
    }

    data = {"meta": json.dumps(meta)}

    t0 = time.time()
    r = requests.post(CLOUD_URL, files=files, data=data)
    latency = (time.time() - t0) * 1000

    try:
        response_json = r.json()
    except:
        print("ERROR: Bad JSON from cloud:", r.text)
        return None, latency

    return response_json, latency


# ---------------------------
# Adaptive scheduling run (one full pass)
# ---------------------------
def run_adaptive_once(target_bw_mbps=5.0):

    frames = sorted(os.listdir(FRAMES_DIR))
    frames = [f for f in frames if f.endswith(".jpg") or f.endswith(".png")]

    frame_records = []
    prev_est_bw = target_bw_mbps
    rounds_total = 0

    for idx, fname in enumerate(frames):
        frame_path = os.path.join(FRAMES_DIR, fname)

        # ---------------------------
        # TIER SELECTION LOGIC
        # (you can plug in your real scheduler later)
        # ---------------------------
        # Simple heuristic:
        # Higher bandwidth → higher tier (1 best … 4 worst)
        if prev_est_bw > 10:
            tier = 1
        elif prev_est_bw > 5:
            tier = 2
        elif prev_est_bw > 2:
            tier = 3
        else:
            tier = 4

        priority, scale, quality = tier_to_params(tier)

        # ---------------------------
        # Prepare compressed frame
        # ---------------------------
        img_bytes = prepare_image(frame_path, scale, quality)
        bytes_len = len(img_bytes)

        # ---------------------------
        # Send to cloud (initial)
        # ---------------------------
        result, latency = send_to_cloud(idx, img_bytes, priority, scale, quality, tier)

        if result is None:
            print("Cloud error – skipping frame", idx)
            continue

        rounds = 1

        # ---------------------------
        # Handle refinement loop
        # ---------------------------
        while result["decision"] == "refine":
            # resend at same tier (or modify if you want)
            img_bytes = prepare_image(frame_path, scale, quality)
            result, latency = send_to_cloud(idx, img_bytes, priority, scale, quality, tier)

            if result is None:
                break

            rounds += 1

        rounds_total += rounds

        # ---------------------------
        # Update estimated bandwidth (very simple model)
        # ---------------------------
        # est_bw = bytes_sent / latency
        est_bw = (bytes_len * 8) / (latency / 1000 + 1e-9) / 1e6
        prev_est_bw = est_bw

        # ---------------------------
        # LOG FRAME ENTRY
        # Every field required by run_exp.py is included.
        # ---------------------------
        frame_records.append({
            "frame": idx,
            "tier": tier,
            "priority": priority,
            "scale": scale,
            "jpeg_quality": quality,
            "bytes_sent": bytes_len,
            "latency": latency,
            "rounds": rounds,
            "est_bw": est_bw,
            "decision": result["decision"],
            "prob": result["prob"]
        })

        print(f"[Frame {idx}] tier={tier}, scale={scale}, q={quality}, "
              f"bytes={bytes_len}, prob={result['prob']:.3f}, "
              f"decision={result['decision']}, rounds={rounds}")

    # ---------------------------
    # Final full log structure
    # REQUIRED BY run_exp.py
    # ---------------------------
    log = {
        "target_bw_mbps": target_bw_mbps,
        "frames_detail": frame_records,
        "avg_latency_ms": sum([x["latency"] for x in frame_records]) / len(frame_records),
        "avg_rounds": sum([x["rounds"] for x in frame_records]) / len(frame_records),
        "total_frames": len(frame_records)
    }

    # Save log
    outpath = f"adaptive_log_{target_bw_mbps}.json"
    with open(outpath, "w") as f:
        json.dump(log, f, indent=4)

    print(f"\n=== Adaptive run for {target_bw_mbps} Mbps saved to {outpath} ===\n")

    return log


# Allow external imports (run_exp.py)
if __name__ == "__main__":
    run_adaptive_once(5.0)
