from __future__ import annotations

import torch
from torch import nn


AGE_GROUPS = [
    "infant",
    "child",
    "teen",
    "young_adult",
    "middle_aged",
    "senior",
]


def _load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """Load state_dict with timm version tolerance.

    Tries strict loading first. If it fails due to architecture changes between
    timm versions (e.g. Swin PatchMerging norm position change), falls back to
    loading only shape-compatible parameters and skipping the rest.
    """
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        own = model.state_dict()
        compatible = {k: v for k, v in state_dict.items()
                      if k in own and own[k].shape == v.shape}
        skipped = len(state_dict) - len(compatible)
        if skipped:
            print(f"  [compat] skipped {skipped} mismatched keys (timm version change)")
        model.load_state_dict(compatible, strict=False)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load the correct model class based on backbone_type stored in checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone_type = ckpt.get("backbone_type", "swin")
    class_names = ckpt["class_names"]
    image_size = ckpt.get("image_size", 224)

    if backbone_type == "vggface":
        model = VGGFaceAgeClassifier(num_classes=len(class_names), pretrained=False).to(device)
    else:
        model = SwinFaceAgeClassifier(
            model_name=ckpt["model_name"],
            num_classes=len(class_names),
            pretrained=False,
        ).to(device)

    _load_state_dict_compat(model, ckpt["model_state_dict"])
    model.eval()
    return model, class_names, image_size, backbone_type


class SwinFaceAgeClassifier(nn.Module):
    """
    Practical SwinFace-inspired age-group classifier.

    This uses a Swin Transformer backbone and a dedicated age classification
    head, which makes it a good first model for age-group experiments.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "timm is required for SwinFaceAgeClassifier. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class EnsembleAgeClassifier(nn.Module):
    """Wraps multiple age classifier models into a single module.

    Forward averages softmax probabilities from each sub-model using
    per-model weights. All models are stored as named submodules so the
    entire ensemble can be saved / loaded as one checkpoint.
    """

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        for i, m in enumerate(models):
            self.add_module(f"model_{i}", m)
        self.num_models = len(models)
        w = torch.tensor(weights if weights else [1.0] * len(models), dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = None
        for i in range(self.num_models):
            probs = torch.softmax(getattr(self, f"model_{i}")(x), dim=1)
            avg = probs * self.weights[i] if avg is None else avg + probs * self.weights[i]
        return avg  # returns averaged probabilities (not logits)


def load_ensemble_from_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load a saved EnsembleAgeClassifier from a single bundled checkpoint file.

    The bundle stores raw individual checkpoint data so each model is restored
    via the same path as load_model_from_checkpoint (timm-version tolerant).
    """
    bundle = torch.load(checkpoint_path, map_location=device)
    class_names = bundle["class_names"]
    weights = bundle.get("weights")
    image_size = 224

    models = []
    for sub_ckpt in bundle["individual_checkpoints"]:
        backbone_type = sub_ckpt.get("backbone_type", "swin")
        cls_names = sub_ckpt["class_names"]
        image_size = sub_ckpt.get("image_size", 224)

        if backbone_type == "vggface":
            m = VGGFaceAgeClassifier(num_classes=len(cls_names), pretrained=False).to(device)
        else:
            m = SwinFaceAgeClassifier(
                model_name=sub_ckpt["model_name"],
                num_classes=len(cls_names),
                pretrained=False,
            ).to(device)

        _load_state_dict_compat(m, sub_ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  Loaded [{backbone_type}] {sub_ckpt.get('model_name', '')}")

    ensemble = EnsembleAgeClassifier(models, weights=weights)
    ensemble.to(device).eval()
    return ensemble, class_names, image_size


class VGGFaceAgeClassifier(nn.Module):
    """Age classifier using VGGFace2-pretrained InceptionResnetV1 backbone.

    Pretrained on 8.6M face images → richer facial feature extraction
    than ImageNet-pretrained models.
    Input: 160x160, normalised to [-1, 1].
    """

    EMBED_DIM = 512

    def __init__(self, num_classes: int = 6, dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "facenet-pytorch is required. Install with: pip install facenet-pytorch"
            ) from exc

        self.backbone = InceptionResnetV1(
            pretrained="vggface2" if pretrained else None,
            classify=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.EMBED_DIM),
            nn.Dropout(dropout),
            nn.Linear(self.EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
