# adaptive_stream.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import time
import heapq
import threading
import numpy as np
import cv2

# -----------------------------
# 1. PriorityFrame
# -----------------------------
@dataclass(order=True)
class PriorityFrame:
    sort_index: float = field(init=False, repr=False)
    priority: float
    frame_id: int
    timestamp: float
    frame: np.ndarray = field(compare=False)
    level: int = field(default=0, compare=False)
    max_level: int = field(default=2, compare=False)
    roi_mask: Optional[np.ndarray] = field(default=None, compare=False)
    meta: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        self.sort_index = -float(self.priority)  # descending

# -----------------------------
# 2. Bandwidth Estimator
# -----------------------------
class BandwidthEstimator:
    def __init__(self, alpha: float = 0.3, initial_bps: float = 1_000_000.0):
        self.alpha = alpha
        self.estimate_bps = initial_bps
        self._lock = threading.Lock()

    def update(self, bytes_sent: int, elapsed_sec: float):
        if elapsed_sec <= 0:
            return
        sample_bps = bytes_sent / elapsed_sec
        with self._lock:
            self.estimate_bps = self.alpha * sample_bps + (1 - self.alpha) * self.estimate_bps

    def get(self) -> float:
        with self._lock:
            return self.estimate_bps

# -----------------------------
# 3. JPEG / WebP Compressor
# -----------------------------
class JPEGCompressor:
    def __init__(self, codec: str = "jpeg"):
        assert codec in ("jpeg", "webp")
        self.codec = codec

    def encode(self, frame: np.ndarray, quality: int = 75, roi_mask: Optional[np.ndarray] = None) -> bytes:
        img = frame

        # ROI-aware mixed-resolution encoding
        if roi_mask is not None:
            downscale = 0.5
            small = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            inv_mask = (roi_mask == 0).astype(np.uint8)
            img = frame.copy()
            img[inv_mask.astype(bool)] = up[inv_mask.astype(bool)]

        # Encode
        if self.codec == "jpeg":
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 5, 95))])
        else:
            ok, buf = cv2.imencode(".webp", img, [int(cv2.IMWRITE_WEBP_QUALITY), int(np.clip(quality, 5, 100))])

        if not ok:
            raise RuntimeError("Encoding failed")
        return buf.tobytes()

# -----------------------------
# 4. Adaptive Scheduler
# -----------------------------
class AdaptiveScheduler:
    def __init__(
        self,
        estimator: BandwidthEstimator,
        compressor: JPEGCompressor,
        low_bps: float = 300_000,
        mid_bps: float = 1_500_000,
        high_bps: float = 5_000_000,
        base_quality: int = 70,
        min_quality: int = 40,
        max_quality: int = 90
    ):
        self.estimator = estimator
        self.compressor = compressor
        self.low_bps = low_bps
        self.mid_bps = mid_bps
        self.high_bps = high_bps
        self.base_quality = base_quality
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.heap: List[PriorityFrame] = []
        self.sent_registry: Dict[Tuple[int, int], bool] = {}

    # enqueue frames
    def enqueue_frames(self, frames: List[PriorityFrame]):
        for f in frames:
            heapq.heappush(self.heap, f)

    # pick appropriate streaming tier
    def _tier_and_quality(self) -> Tuple[str, int]:
        bps = self.estimator.get()
        if bps < self.low_bps:
            return "low", max(self.min_quality, self.base_quality - 20)
        elif bps < self.mid_bps:
            return "mid", self.base_quality
        elif bps < self.high_bps:
            return "high", min(self.max_quality, self.base_quality + 10)
        else:
            return "ultra", min(self.max_quality, self.base_quality + 15)

    # identify next best frame to transmit
    def pick_next(self) -> Optional[PriorityFrame]:
        if not self.heap:
            return None
        
        best = None
        temp = []

        while self.heap:
            f = heapq.heappop(self.heap)
            if (f.frame_id, f.level) not in self.sent_registry:
                best = f
                break
            temp.append(f)

        for f in temp:
            heapq.heappush(self.heap, f)

        return best

    # perform a single transmission cycle
    def transmit_once(self) -> Optional[Dict[str, Any]]:
        f = self.pick_next()
        if f is None:
            return None

        tier, quality = self._tier_and_quality()

        level = f.level
        scale_map = {0: 0.5, 1: 0.75, 2: 1.0}
        scale = scale_map.get(level, 1.0)

        frame = f.frame
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        roi_mask = f.roi_mask if level == f.max_level else None

        t0 = time.time()
        payload = self.compressor.encode(resized, quality=quality, roi_mask=roi_mask)
        elapsed = (time.time() - t0)
        self.estimator.update(len(payload), elapsed)

        self.sent_registry[(f.frame_id, level)] = True

        return {
            "frame_id": f.frame_id,
            "tier": tier,
            "quality": quality,
            "level": level,
            "scale": scale,
            "bytes": len(payload),
            "elapsed_sec": elapsed,
            "est_bps_after": self.estimator.get(),
            "priority": f.priority,
            "payload_bytes": payload,
            "mime": "image/jpeg" if self.compressor.codec=="jpeg" else "image/webp",
        }
