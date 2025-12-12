# Adaptive Telemedicine Edge–Cloud Streaming (CSE 589 Project)

This repository contains a prototype edge–cloud streaming system for telemedicine imaging under constrained and fluctuating bandwidth. The codebase includes:

- A more realistic **edge-side scheduler and orchestrator** (`adaptive_stream.py`, `edge_orchestrator.py`).
- Simple **cloud and edge server stubs** for HTTP-based interaction.

The main goal is to simulate and evaluate bandwidth-adaptive, priority-aware streaming of medical frames.

---

## 1. Repository Structure

Key files (only the most relevant ones):
- `adaptive_runner.py` – Simulates adaptive streaming against a simple cloud endpoint.
- `edge_tiers.py` – Bandwidth estimator and tier helper functions for experiments.
- `adaptive_stream.py` – Core adaptive scheduler:
  - `PriorityFrame`, `BandwidthEstimator`, `JPEGCompressor`, `AdaptiveScheduler`.
- `edge_orchestrator.py` – Edge-side orchestrator that:
  - Loads frames, runs edge inference, and uses `AdaptiveScheduler` to send to a cloud endpoint.
- `edge_server.py` – FastAPI-based edge server that wraps the adaptive scheduler.
- `cloud_server.py` – Simple FastAPI cloud stub for `adaptive_runner` / `run_exp.py`.
- `cloud_model.py` – Cloud-side model definition (standalone; not required to run the basic stub).

---

## 2. Environment Setup

### 2.1. Python Version

- Python **3.9+** is recommended.

### 2.2. Dependencies

Install via `pip` (in a virtualenv or conda environment is recommended):

```bash
pip install \
    numpy \
    matplotlib \
    pillow \
    opencv-python \
    requests \
    fastapi \
    "uvicorn[standard]" \
    torch
```

## 3. Edge-Cloud Adaptive Streming
### Start cloud server
```bash
python cloud_server.py
```
### Start edge server
```bash
python edge_server.py
```
### Run Edge-Cloud Adpative Streaming
```bash
python edge_orchestrator.py \
    --frames_dir frames \
    --server http://localhost:9000/upload_frame \
    --model edge_telemed_int8.pt \
    --device cpu \
    --initial_mbps 1.0 \
    --codec jpeg
```
