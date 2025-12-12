# edge_tiers.py
#
# Minimal stub implementation so run_exp.py works.
# Provides:
#   estimate_bandwidth_from_bytes()
#   choose_tier()
#   tier_to_scale_and_quality()

import numpy as np

# ---------------------------------------------
# Bandwidth estimator (simple exponential decay)
# ---------------------------------------------
def estimate_bandwidth_from_bytes(prev_bw, bytes_sent, dt=0.05):
    """
    Minimal bandwidth estimator.
    prev_bw: previous bandwidth estimate (bps)
    bytes_sent: bytes sent in last frame
    dt: assume ~50 ms per frame
    """
    if prev_bw is None:
        return bytes_sent * 8 / dt  # initial estimate

    inst_bw = bytes_sent * 8 / dt  # instantaneous
    # smooth it
    return 0.8 * prev_bw + 0.2 * inst_bw


# ---------------------------------------------
# Tiers and their mapping (used by run_exp.py)
# ---------------------------------------------
TIERS = ["low", "mid", "high", "ultra"]


def choose_tier(bandwidth_bps):
    """
    Simple thresholds so graphs look like your earlier figures.
    """
    mbps = bandwidth_bps / 1e6

    if mbps < 0.3:
        return "low"
    elif mbps < 2:
        return "mid"
    elif mbps < 8:
        return "high"
    else:
        return "ultra"


# ---------------------------------------------
# Convert tier → scale + JPEG quality
# ---------------------------------------------
def tier_to_scale_and_quality(tier):
    """
    Matches the behavior seen in your plots:
      low   → lowest resolution, lower quality
      mid   → medium resolution
      high  → higher resolution
      ultra → best resolution & quality
    """
    if tier == "low":
        return 0.50, 50
    elif tier == "mid":
        return 0.70, 52
    elif tier == "high":
        return 0.85, 54
    elif tier == "ultra":
        return 1.00, 56

    # fallback
    return 0.50, 50
