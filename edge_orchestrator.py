# edge_orchestrator.py
import argparse, os, json, time
import numpy as np
import cv2
import torch
import requests

from adaptive_stream import PriorityFrame, BandwidthEstimator, JPEGCompressor, AdaptiveScheduler

# -----------------------------
# 1. Edge Inference Runner
# -----------------------------
class EdgeModelRunner:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            self.heuristic_mode = False
        except:
            print("[edge] WARN: model not loaded; using heuristic scoring.")
            self.model = None
            self.heuristic_mode = True

    @torch.inference_mode()
    def infer(self, img_bgr):
        if self.heuristic_mode:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            thr = np.percentile(gray, 90)
            roi = (gray >= thr).astype(np.uint8)
            area = roi.sum() / roi.size
            contrast = (gray.max() - gray.min()) / 255.0
            score = 70 * area + 30 * contrast
            return score, roi

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
        out = self.model(ten)

        score = float(torch.sigmoid(out["score"]).item() * 100.0) if isinstance(out, dict) else 50.0
        mask  = (torch.sigmoid(out["mask"][0,0]) > 0.5).cpu().numpy().astype(np.uint8) if "mask" in out else None
        
        return score, mask

# -----------------------------
# 2. File Loader
# -----------------------------
def list_frames(folder):
    return sorted([os.path.join(folder,f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg",".png",".jpeg",".bmp",".tif",".tiff"))])

# -----------------------------
# 3. Main Loop
# -----------------------------
def edge_loop(folder, server, model_path, device, initial_mbps, codec):
    runner = EdgeModelRunner(model_path, device)
    estimator = BandwidthEstimator(initial_bps=initial_mbps * 1_000_000)
    compressor = JPEGCompressor(codec)
    sched = AdaptiveScheduler(estimator, compressor)

    paths = list_frames(folder)
    buf = []

    print(f"[edge] Loading frames from: {folder}")
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        score, roi = runner.infer(img)
        buf.append(PriorityFrame(priority=score, frame_id=i, timestamp=time.time(),
                                 frame=img, roi_mask=roi, max_level=2))
    sched.enqueue_frames(buf)

    print("[edge] Begin adaptive transmission…")

    while True:
        rec = sched.transmit_once()
        if rec is None:
            print("[edge] All frames transmitted")
            break

        print(f"[edge] tx frame={rec['frame_id']:04d} lvl={rec['level']} "
              f"tier={rec['tier']} q={rec['quality']} bytes={rec['bytes']} "
              f"estbps={rec['est_bps_after']:.1f}")

        if server:
            files = {"frame": ("frame.jpg", rec["payload_bytes"], rec["mime"])}
            metadata = json.dumps({k:v for k,v in rec.items() if k!='payload_bytes'})

            try:
                r = requests.post(server, files=files, data={"meta": metadata}, timeout=5)
                r.raise_for_status()
                resp = r.json()

                # If cloud requests more frames at higher quality
                for req in resp.get("next_requests", []):
                    fid  = req.get("frame_id")
                    bump = buf[fid]
                    bump2 = PriorityFrame(priority=bump.priority + 10.0,
                                          frame_id=bump.frame_id,
                                          timestamp=time.time(),
                                          frame=bump.frame,
                                          roi_mask=bump.roi_mask,
                                          max_level=bump.max_level)
                    sched.enqueue_frames([bump2])

            except Exception as e:
                print("[edge] Upload failed:", e)

    print("[edge] Done.")

# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--server", default="http://localhost:9000/upload_frame")
    ap.add_argument("--model", default="edge_telemed_int8.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--initial_mbps", type=float, default=1.0)
    ap.add_argument("--codec", choices=["jpeg","webp"], default="jpeg")
    args = ap.parse_args()

    edge_loop(args.frames_dir, args.server, args.model, args.device, args.initial_mbps, args.codec)

if __name__ == "__main__":
    main()
