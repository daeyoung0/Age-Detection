"""Non-interactive smoke test for the webcam inference pipeline.

Loads the same models as camera_infer_swin.py, grabs a bounded number of
webcam frames (no cv2.imshow, no infinite loop), runs detection + age +
gender on each, and posts to the backend. Exits automatically.
"""
import sys
import time

import torch

import camera_infer_swin as cam

BACKEND = "http://127.0.0.1:8000"
MAX_FRAMES = 40
MAX_SECONDS = 20.0


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device: {device}")

    print("[smoke] loading RetinaFace...")
    from retinaface.pre_trained_models import get_model
    face_model = get_model("resnet50_2020-07-20", max_size=1024, device=device)
    face_model.eval()

    print("[smoke] loading age ensemble (ensemble_korean.pt)...")
    try:
        age_model, class_names, image_size = cam.load_ensemble_model("ensemble_korean.pt", device)
    except ModuleNotFoundError as exc:
        print(f"[smoke] ENSEMBLE LOAD FAILED (missing dep): {exc}")
        return 2
    print(f"[smoke] classes: {class_names} | image_size: {image_size}")

    from src.swinface_age.dataset import build_eval_transforms
    transforms = [build_eval_transforms(image_size=image_size)]  # --disable-tta

    print("[smoke] loading FairFace gender model...")
    gender_model = None
    try:
        gender_model = cam.create_gender_classifier("fairface", device)
    except Exception as exc:
        print(f"[smoke] gender unavailable: {exc}")

    print("[smoke] opening webcam index 0...")
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[smoke] FAILED to open webcam")
        return 3

    start = time.time()
    frames = 0
    ok_faces = 0
    posted = 0
    while frames < MAX_FRAMES and (time.time() - start) < MAX_SECONDS:
        ok, frame = cap.read()
        frames += 1
        if not ok:
            continue
        result = cam.predict_face_result(
            frame, face_model, age_model, class_names, transforms, device,
            face_margin=0.15, min_score=0.8, min_size=80, use_clahe=False,
        )
        if result.get("status") != "ok":
            continue
        if cam.low_quality_face(result, min_face=80):
            print(f"[frame {frames}] face found but low quality -> skipped")
            continue
        ok_faces += 1
        gender_label = "unknown"
        gender_age = None
        if gender_model is not None:
            gi = result.get("aligned_crop") or result["crop"]
            gr = gender_model.predict(gi)
            gender_label, gender_age = gr.gender, gr.source_age
        biased = cam.apply_age_bias(list(result["all_probs"].values()), gender_label, list(result["all_probs"].keys()))
        names = list(result["all_probs"].keys())
        si = max(range(len(names)), key=lambda i: biased[i])
        age_group, conf = names[si], biased[si]
        x1, y1, x2, y2 = result["bbox"]
        print(f"[frame {frames}] bbox={x2-x1}x{y2-y1} | age={age_group} ({conf:.2f}) | gender={gender_label} age~{gender_age}")
        cam.post_to_backend(BACKEND, age_group, result["face_size"], gender_label)
        posted += 1

    cap.release()
    print(f"[smoke] done. frames={frames} ok_faces={ok_faces} posted={posted}")
    if ok_faces == 0:
        print("[smoke] NOTE: no usable face detected (camera may show empty scene / poor lighting).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
