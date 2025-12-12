# edge_client.py
import os
import json
import time
import random
import requests

CLOUD_URL = "http://127.0.0.1:8001/submit_frame"

def edge_confidence(frame_id):
    """Fake edge model confidence"""
    return random.uniform(0, 1)

def send_to_cloud(frame_path, meta):
    files = {"frame": open(frame_path, "rb")}
    data = {"meta": json.dumps(meta)}

    r = requests.post(CLOUD_URL, files=files, data=data)

    print("RAW RESPONSE:", r.status_code, r.text)   # <--- ADD THIS

    try:
        return r.json()
    except Exception:
        print("ERROR parsing JSON")
        return {"error": r.text}


def process_frame(frame_id, frame_path):
    logs = {"frame_id": frame_id, "rounds": 0}

    edge_score = edge_confidence(frame_id)
    logs["edge_score"] = edge_score

    if edge_score > 0.75:
        logs["decision"] = "edge_accept"
        return logs

    # send to cloud
    meta = {"frame_id": frame_id, "compression": 1}
    result = send_to_cloud(frame_path, meta)
    logs["cloud_round_1"] = result
    logs["rounds"] += 1

    # refinement loop
    while result["decision"] == "refine":
        meta["compression"] += 1
        result = send_to_cloud(frame_path, meta)
        logs[f"cloud_round_{logs['rounds']+1}"] = result
        logs["rounds"] += 1

    logs["final_cloud_decision"] = result["decision"]
    return logs
