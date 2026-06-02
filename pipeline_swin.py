"""
Age group prediction pipeline using ensemble Swin/ConvNeXt models + RetinaFace detection.

Single file usage:
    from pipeline_swin import load_ensemble, predict_age_group

    pipeline = load_ensemble("ensemble_best.pt")
    result = pipeline.predict("image.jpg")
"""
from __future__ import annotations

import cv2
import torch
from PIL import Image

from src.swinface_age.dataset import build_eval_transforms, build_tta_transforms
from src.swinface_age.model import load_ensemble_from_checkpoint


class AgePipeline:
    def __init__(self, ensemble, class_names: list[str], image_size: int, device: torch.device, tta: bool = True):
        self.ensemble = ensemble
        self.class_names = class_names
        self.image_size = image_size
        self.device = device
        self.tta = tta
        self.transforms = build_tta_transforms(image_size) if tta else [build_eval_transforms(image_size)]

        from retinaface.pre_trained_models import get_model
        self.face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
        self.face_detector.eval()

    def predict(self, img_path: str) -> list[dict]:
        """
        Detect faces and predict age group for each face.

        Returns list of dicts:
            [{"bbox": [x1,y1,x2,y2], "age_group": "young_adult", "confidence": 0.91, "probs": {...}}]
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotations = self.face_detector.predict_jsons(img_rgb)

        results = []
        for ann in annotations:
            if ann.get("bbox") is None or len(ann["bbox"]) < 4:
                continue
            x1, y1, x2, y2 = map(int, ann["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            face_crop = Image.fromarray(img_rgb[y1:y2, x1:x2])
            probs = self._classify(face_crop)

            pred_idx = probs.argmax().item()
            results.append({
                "bbox": [x1, y1, x2, y2],
                "age_group": self.class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probs": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
            })

        return results

    def predict_crop(self, face_image) -> dict:
        """
        Predict age group from an already-cropped face image (PIL Image or numpy BGR array).

        Returns dict: {"age_group": "young_adult", "confidence": 0.91, "probs": {...}}
        """
        if not isinstance(face_image, Image.Image):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

        probs = self._classify(face_image)
        pred_idx = probs.argmax().item()
        return {
            "age_group": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
        }

    @torch.no_grad()
    def _classify(self, face_pil: Image.Image) -> torch.Tensor:
        avg = None
        for t in self.transforms:
            tensor = t(face_pil).unsqueeze(0).to(self.device)
            probs = self.ensemble(tensor)[0]
            avg = probs if avg is None else avg + probs
        return avg / len(self.transforms)


def load_ensemble(checkpoint_path: str, device: torch.device | None = None, tta: bool = True) -> AgePipeline:
    """Load ensemble model and return AgePipeline ready for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble, class_names, image_size = load_ensemble_from_checkpoint(checkpoint_path, device)
    return AgePipeline(ensemble, class_names, image_size, device, tta=tta)


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "ensemble_best.pt"
    img_path = sys.argv[2] if len(sys.argv) > 2 else "images/test.jpg"

    print(f"Loading ensemble from {checkpoint} ...")
    pipeline = load_ensemble(checkpoint)
    print(f"Running on {img_path} ...")
    results = pipeline.predict(img_path)

    if not results:
        print("No faces detected.")
    for i, r in enumerate(results):
        print(f"Face {i+1}: {r['age_group']} (confidence: {r['confidence']:.4f})")
        for name, p in r["probs"].items():
            print(f"  {name:15s} {p:.4f}")
