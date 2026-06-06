import argparse
import json
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
from PIL import Image
import torch

from retinaface.pre_trained_models import get_model

from src.swinface_age.dataset import build_eval_transforms, build_tta_transforms
from src.swinface_age.model import load_ensemble_from_checkpoint

DEFAULT_ENSEMBLE_CHECKPOINT = str(Path(__file__).resolve().with_name("ensemble_korean.pt"))

# Example camera URLs:
#   http://192.168.219.101:8050/video
#   rtsp://192.168.0.10:554/stream1


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face and age-group inference with ensemble model")
    parser.add_argument("--camera-url", type=str, default="", help="IP camera stream URL, e.g. http://192.168.219.101:8050/video")
    parser.add_argument("--camera-index", type=int, default=0, help="Local camera index")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_ENSEMBLE_CHECKPOINT, help="Ensemble model checkpoint")
    parser.add_argument("--max-size", type=int, default=1024, help="RetinaFace max size")
    parser.add_argument("--face-margin", type=float, default=0.15, help="Fraction to expand the detected face bbox. Default 0.15 matches the Korean fine-tune training crop (finetune_korean_swin.py). 0 = tight RetinaFace box.")
    parser.add_argument("--disable-tta", action="store_true", help="Disable test-time augmentation for ensemble inference")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000", help="Backend server URL")
    parser.add_argument("--post-interval", type=float, default=1.0, help="Seconds between backend POSTs (default: 1.0)")
    parser.add_argument("--debug", action="store_true", help="Print full per-class probabilities and save the face crop the model actually sees to debug_crops/")
    parser.add_argument("--min-score", type=float, default=0.8, help="Minimum RetinaFace detection score to accept a face")
    parser.add_argument("--min-face", type=int, default=80, help="Minimum face bbox size (px) to accept")
    parser.add_argument("--smooth", type=int, default=8, help="Number of recent frames to average predictions over (1 = no smoothing)")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast boost to the crop (off by default; this is a train/inference mismatch, A/B test it)")
    return parser.parse_args()


def load_ensemble_model(checkpoint_path, device):
    ensemble, class_names, image_size = load_ensemble_from_checkpoint(checkpoint_path, device)
    return ensemble, class_names, image_size


def open_capture(args):
    if args.camera_url:
        cap = cv2.VideoCapture(args.camera_url)
        source_name = args.camera_url
    else:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera_index)
        source_name = f"index={args.camera_index}"

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source_name}")
    return cap, source_name


def apply_clahe(rgb):
    """Contrast-limited adaptive histogram equalization on the L channel.
    WARNING: training data was NOT CLAHE-processed, so enabling this at inference
    is a deliberate train/inference mismatch — A/B test it, don't assume it helps."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def pick_largest_face(annotations, min_score=0.8, min_size=80, max_aspect=1.5):
    """Pick the largest *valid* face. Rejects weak detections, tiny boxes, and
    non-face-shaped boxes (very wide strips) that otherwise collapse the age
    model to infant/child."""
    valid = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        if ann.get("score", 1.0) < min_score:
            continue
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < min_size or h < min_size:
            continue
        if h <= 0 or w / h > max_aspect:  # wider than a real upright face -> bad crop
            continue
        valid.append(ann)
    if not valid:
        return None
    return max(valid, key=lambda ann: (ann["bbox"][2] - ann["bbox"][0]) * (ann["bbox"][3] - ann["bbox"][1]))


@torch.no_grad()
def classify_face(face_image, age_model, class_names, transforms, device):
    avg = None
    for transform in transforms:
        input_tensor = transform(face_image).unsqueeze(0).to(device)
        probs = age_model(input_tensor)[0]
        avg = probs if avg is None else avg + probs
    probs = avg / len(transforms)
    pred_idx = probs.argmax().item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs = {class_names[i]: probs[i].item() for i in range(len(class_names))}
    return pred_class, confidence, all_probs


def predict_face_result(frame_bgr, face_model, age_model, class_names, transforms, device,
                        face_margin=0.15, min_score=0.8, min_size=80, use_clahe=False):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    annotations = face_model.predict_jsons(frame_rgb)
    if not annotations:
        return {"status": "no_detections", "detections": 0}

    face = pick_largest_face(annotations, min_score=min_score, min_size=min_size)
    if face is None:
        return {"status": "invalid_detections", "detections": len(annotations)}

    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, face["bbox"])
    # Expand the tight RetinaFace box so the crop framing (hair/forehead/chin)
    # matches the loosely-cropped training data, which reduces the youthful bias.
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * face_margin), int(bh * face_margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return {"status": "face_too_small", "detections": len(annotations)}

    crop_rgb = frame_rgb[y1:y2, x1:x2]
    if use_clahe:
        crop_rgb = apply_clahe(crop_rgb)
    face_crop = Image.fromarray(crop_rgb)
    pred_class, confidence, all_probs = classify_face(face_crop, age_model, class_names, transforms, device)
    label = f"{pred_class} ({confidence:.2f})"

    return {
        "status": "ok",
        "bbox": (x1, y1, x2, y2),
        "age_group": pred_class,
        "all_probs": all_probs,
        "crop": face_crop,
        "face_size": (x2 - x1) * (y2 - y1),
        "label": label,
        "detections": len(annotations),
    }


def _post_worker(backend_url: str, age_group: str, face_size: int):
    try:
        body = json.dumps({"age_group": age_group, "face_size": face_size}).encode()
        req = urllib.request.Request(
            f"{backend_url}/api/camera",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2):
            pass
    except Exception:
        pass


def post_to_backend(backend_url: str, age_group: str, face_size: int):
    threading.Thread(target=_post_worker, args=(backend_url, age_group, face_size), daemon=True).start()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading RetinaFace...")
    face_model = get_model("resnet50_2020-07-20", max_size=args.max_size, device=device)
    face_model.eval()

    print(f"Loading ensemble model: {args.checkpoint}")
    age_model, class_names, image_size = load_ensemble_model(args.checkpoint, device)
    transforms = [build_eval_transforms(image_size=image_size)] if args.disable_tta else build_tta_transforms(image_size=image_size)

    cap, source_name = open_capture(args)
    print(f"Camera opened: {source_name}")
    print(f"Backend: {args.backend_url}  |  POST interval: {args.post_interval}s")
    print("Press 'q' to quit.")

    last_post_time = 0.0
    debug_idx = 0
    prob_hist = deque(maxlen=max(1, args.smooth))  # recent per-class prob vectors
    miss = 0
    if args.debug:
        Path("debug_crops").mkdir(exist_ok=True)
        print(f"[debug] margin={args.face_margin}  tta={'off' if args.disable_tta else 'on'}  smooth={args.smooth}  crops -> debug_crops/")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        result = predict_face_result(frame, face_model, age_model, class_names, transforms, device,
                                     face_margin=args.face_margin, min_score=args.min_score, min_size=args.min_face,
                                     use_clahe=args.clahe)

        if result is not None and result.get("status") == "ok":
            x1, y1, x2, y2 = result["bbox"]

            # temporal smoothing: average per-class probs over recent frames so a
            # single bad-detection frame can't flip the label to infant/child
            miss = 0
            prob_hist.append([result["all_probs"][c] for c in class_names])
            mean = [sum(v[i] for v in prob_hist) / len(prob_hist) for i in range(len(class_names))]
            si = max(range(len(class_names)), key=lambda i: mean[i])
            age_group, confidence = class_names[si], mean[si]
            label = f"{age_group} ({confidence:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            now = time.monotonic()
            if now - last_post_time >= args.post_interval:
                post_to_backend(args.backend_url, age_group, result["face_size"])
                last_post_time = now
                cv2.putText(frame, "SENT", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

                if args.debug:
                    raw = sorted(result["all_probs"].items(), key=lambda kv: kv[1], reverse=True)
                    dist = "  ".join(f"{k}={v:.2f}" for k, v in raw)
                    print(f"[debug] bbox={x2-x1}x{y2-y1}px | smoothed={age_group} {confidence:.2f} | raw {dist}")
                    crop_path = f"debug_crops/crop_{debug_idx:03d}_{result['age_group']}.jpg"
                    result["crop"].save(crop_path)
                    debug_idx += 1
        else:
            # lost the face for several frames -> drop stale history so the next
            # person doesn't inherit the previous person's smoothed prediction
            miss += 1
            if miss >= 5:
                prob_hist.clear()
            status = "unknown" if result is None else result.get("status", "unknown")
            detections = 0 if result is None else result.get("detections", 0)
            cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Detections: {detections}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Real-time Age Inference (Ensemble)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
