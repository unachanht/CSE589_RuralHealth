# cloud_server.py
import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import json
import numpy as np

app = FastAPI()

@app.post("/submit_frame")
async def submit_frame(frame: UploadFile, meta: str = Form(...)):

    # read bytes safely
    img_bytes = await frame.read()

    # dummy score
    score = float(np.random.rand())

    # adaptive logic
    if 0.3 < score < 0.7:
        decision = "refine"
        next_req = [{"frame_id": json.loads(meta)["frame_id"], "reason": "uncertain"}]
    else:
        decision = "ok"
        next_req = []

    return JSONResponse({
        "decision": decision,
        "prob": score,
        "next_requests": next_req
    })

@app.get("/status")
async def status():
    return {"status": "cloud server ok"}

if __name__ == "__main__":
    uvicorn.run("cloud_server:app", host="127.0.0.1", port=8001)
