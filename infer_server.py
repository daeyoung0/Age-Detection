"""GPU inference API server — runs on RunPod (or any CUDA box).

Loads the heavy models (RetinaFace + age ensemble + FairFace gender) once,
then serves single-frame inference over HTTP. The local machine captures the
camera and displays the window; only inference runs here.

Run on RunPod:
    python -m uvicorn infer_server:app --host 0.0.0.0 --port 8000

Then expose port 8000 via RunPod's HTTP proxy and point camera_client.py at
    https://<POD_ID>-8000.proxy.runpod.net/infer

Endpoints:
    GET  /health  -> {"device": "cuda", "cuda": true, "classes": [...]}
    POST /infer   -> body = raw JPEG/PNG bytes (Content-Type: image/jpeg)
                     returns the final smoothed-per-call prediction for the
                     largest valid face, or a status if none.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request

import camera_infer_swin as cam
from src.swinface_age.dataset import build_eval_transforms, build_tta_transforms

CHECKPOINT = os.environ.get("ENSEMBLE_CHECKPOINT", "ensemble_korean.pt")
MAX_SIZE = int(os.environ.get("RETINAFACE_MAX_SIZE", "1024"))
USE_TTA = os.environ.get("USE_TTA", "0") == "1"
DISABLE_GENDER = os.environ.get("DISABLE_GENDER", "0") == "1"

app = FastAPI(title="Age/Gender GPU Inference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[infer] device: {device}")

print("[infer] loading RetinaFace...")
from retinaface.pre_trained_models import get_model
face_model = get_model("resnet50_2020-07-20", max_size=MAX_SIZE, device=device)
face_model.eval()

print(f"[infer] loading age ensemble: {CHECKPOINT}")
age_model, class_names, image_size = cam.load_ensemble_model(CHECKPOINT, device)
transforms = build_tta_transforms(image_size) if USE_TTA else [build_eval_transforms(image_size)]
print(f"[infer] classes: {class_names}")

gender_model = None
if not DISABLE_GENDER:
    try:
        gender_model = cam.create_gender_classifier("fairface", device)
        print("[infer] FairFace gender loaded")
    except Exception as exc:  # noqa: BLE001
        print(f"[infer] gender unavailable: {exc}")


@app.get("/health")
def health():
    return {
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "classes": class_names,
        "gender": gender_model is not None,
    }


@app.post("/infer")
async def infer(request: Request):
    raw = await request.body()
    if not raw:
        return {"status": "empty_body"}

    arr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if frame is None:
        return {"status": "decode_failed"}

    result = cam.predict_face_result(
        frame, face_model, age_model, class_names, transforms, device,
        face_margin=0.15, min_score=0.8, min_size=80, use_clahe=False,
    )
    if result.get("status") != "ok":
        return {"status": result.get("status", "unknown"), "detections": result.get("detections", 0)}
    if cam.low_quality_face(result, min_face=80):
        return {"status": "low_quality", "bbox": list(result["bbox"])}

    gender_label = "unknown"
    gender_age = None
    if gender_model is not None:
        gi = result.get("aligned_crop") if result.get("aligned_crop") is not None else result["crop"]
        gr = gender_model.predict(gi)
        gender_label, gender_age = gr.gender, gr.source_age

    names = list(result["all_probs"].keys())
    biased = cam.apply_age_bias(list(result["all_probs"].values()), gender_label, names)
    si = max(range(len(names)), key=lambda i: biased[i])

    return {
        "status": "ok",
        "bbox": list(result["bbox"]),
        "age_group": names[si],
        "confidence": float(biased[si]),
        "gender": gender_label,
        "gender_age": gender_age,
        "face_size": int(result["face_size"]),
        "probs": {names[i]: float(biased[i]) for i in range(len(names))},
    }
