from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


FAIRFACE_AGE_BUCKETS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]

FAIRFACE_AGE_HINTS = {
    "0-2": 1,
    "3-9": 6,
    "10-19": 15,
    "20-29": 25,
    "30-39": 35,
    "40-49": 45,
    "50-59": 55,
    "60-69": 65,
    "70+": 75,
}


@dataclass
class GenderPrediction:
    gender: str
    matched: bool
    source_age: int | None = None
    match_score: float = 0.0


class FairFaceGenderClassifier:
    """Crop-based FairFace gender classifier.

    Uses the official FairFace ResNet-34 checkpoint and reads the gender logits
    from indices 7:9, matching the official inference scripts.
    """

    def __init__(self, weights_path: str | Path, device: torch.device) -> None:
        self.device = device
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"FairFace weights not found: {self.weights_path}")

        self.model = models.resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        state_dict = torch.load(self.weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, face_crop) -> GenderPrediction:
        if face_crop is None:
            return GenderPrediction(gender="unknown", matched=False, source_age=None, match_score=0.0)

        if not isinstance(face_crop, Image.Image):
            face_crop = Image.fromarray(np.asarray(face_crop))
        face_crop = face_crop.convert("RGB")

        tensor = self.transform(face_crop).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)[0]

        gender_logits = outputs[7:9]
        age_logits = outputs[9:18]
        gender_probs = torch.softmax(gender_logits, dim=0)
        age_probs = torch.softmax(age_logits, dim=0)

        gender_idx = int(torch.argmax(gender_probs).item())
        age_idx = int(torch.argmax(age_probs).item())

        label = "male" if gender_idx == 0 else "female"
        age_bucket = FAIRFACE_AGE_BUCKETS[age_idx]
        age_value = FAIRFACE_AGE_HINTS[age_bucket]
        score = float(gender_probs[gender_idx].item())
        return GenderPrediction(gender=label, matched=True, source_age=age_value, match_score=score)


def create_gender_classifier(model_name: str, device: torch.device):
    if model_name == "fairface":
        weights_path = Path("fair_face_models") / "res34_fair_align_multi_7_20190809.pt"
        return FairFaceGenderClassifier(weights_path=weights_path, device=device)
    raise ValueError(f"Unsupported gender model: {model_name}")
