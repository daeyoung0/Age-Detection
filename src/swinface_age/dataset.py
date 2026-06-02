from __future__ import annotations

from pathlib import Path
import re
import csv

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import datasets, transforms


IMAGE_SIZE = 224
AGE_GROUPS = [
    "infant",
    "child",
    "teen",
    "young_adult",
    "middle_aged",
    "senior",
]


def age_to_group(age: int) -> str:
    if 0 <= age <= 5:
        return "infant"
    if 6 <= age <= 12:
        return "child"
    if 13 <= age <= 18:
        return "teen"
    if 19 <= age <= 34:
        return "young_adult"
    if 35 <= age <= 49:
        return "middle_aged"
    return "senior"


def parse_age_from_morph_filename(filename: str) -> int:
    match = re.search(r"[MF](\d+)", filename, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse age from filename: {filename}")
    return int(match.group(1))


class MorphAgeDataset(Dataset):
    def __init__(self, image_dir: str | Path, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_names = AGE_GROUPS
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []

        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        for path in sorted(self.image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in valid_suffixes:
                continue
            age = parse_age_from_morph_filename(path.name)
            age_group = age_to_group(age)
            label = self.class_to_idx[age_group]
            self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No image files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class OrderedImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):
        classes, class_to_idx = super().find_classes(directory)
        if set(classes) != set(AGE_GROUPS):
            missing = sorted(set(AGE_GROUPS) - set(classes))
            extra = sorted(set(classes) - set(AGE_GROUPS))
            raise ValueError(
                "ImageFolder classes do not match expected age groups. "
                f"Missing: {missing}, Extra: {extra}"
            )

        ordered_classes = AGE_GROUPS[:]
        ordered_class_to_idx = {
            class_name: idx for idx, class_name in enumerate(ordered_classes)
        }
        return ordered_classes, ordered_class_to_idx


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        split: str,
        transform=None,
    ) -> None:
        self.transform = transform
        self.class_names = AGE_GROUPS
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[Path, int]] = []

        for row in rows:
            if row["split"] != split:
                continue
            class_name = row["class_name"]
            if class_name not in self.class_to_idx:
                continue
            image_path = Path(row["saved_path"])
            if not image_path.exists():
                continue
            self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in manifest dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            transforms.RandomErasing(
                p=0.2,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def build_tta_transforms(image_size: int = IMAGE_SIZE) -> list:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return [
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(int(image_size * 1.15)), transforms.CenterCrop(image_size), transforms.ToTensor(), normalize]),
    ]


VGGFACE_IMAGE_SIZE = 160


def build_vggface_train_transforms(image_size: int = VGGFACE_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])


def build_vggface_eval_transforms(image_size: int = VGGFACE_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_vggface_tta_transforms(image_size: int = VGGFACE_IMAGE_SIZE) -> list:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return [
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(int(image_size * 1.15)), transforms.CenterCrop(image_size), transforms.ToTensor(), normalize]),
    ]


def build_eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_datasets(data_dir: str | Path, image_size: int = IMAGE_SIZE):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = OrderedImageFolder(
        train_dir, transform=build_train_transforms(image_size)
    )
    val_dataset = OrderedImageFolder(
        val_dir, transform=build_eval_transforms(image_size)
    )

    return train_dataset, val_dataset


def build_morph_datasets(data_dir: str | Path, image_size: int = IMAGE_SIZE):
    data_dir = Path(data_dir)
    images_dir = data_dir / "Images"
    train_dir = images_dir / "Train"
    val_dir = images_dir / "Validation"

    train_dataset = MorphAgeDataset(
        train_dir, transform=build_train_transforms(image_size)
    )
    val_dataset = MorphAgeDataset(
        val_dir, transform=build_eval_transforms(image_size)
    )

    return train_dataset, val_dataset


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def build_manifest_datasets(
    manifest_path: str | Path,
    image_size: int = IMAGE_SIZE,
):
    rows = load_manifest_rows(manifest_path)
    train_dataset = ManifestImageDataset(
        rows,
        split="train",
        transform=build_train_transforms(image_size),
    )
    val_dataset = ManifestImageDataset(
        rows,
        split="val",
        transform=build_eval_transforms(image_size),
    )
    return train_dataset, val_dataset


def build_morph_test_dataset(data_dir: str | Path, image_size: int = IMAGE_SIZE):
    data_dir = Path(data_dir)
    test_dir = data_dir / "Images" / "Test"
    if not test_dir.exists():
        return None

    return MorphAgeDataset(test_dir, transform=build_eval_transforms(image_size))


def build_test_dataset(data_dir: str | Path, image_size: int = IMAGE_SIZE):
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    if not test_dir.exists():
        return None

    return OrderedImageFolder(test_dir, transform=build_eval_transforms(image_size))


def build_manifest_test_dataset(
    manifest_path: str | Path,
    image_size: int = IMAGE_SIZE,
):
    rows = load_manifest_rows(manifest_path)
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        return None
    return ManifestImageDataset(
        rows,
        split="test",
        transform=build_eval_transforms(image_size),
    )


def build_test_loader(
    data_dir: str | Path,
    dataset_type: str = "imagefolder",
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
    num_workers: int = 4,
):
    if dataset_type == "morph":
        test_dataset = build_morph_test_dataset(data_dir, image_size=image_size)
    elif dataset_type == "manifest":
        test_dataset = build_manifest_test_dataset(data_dir, image_size=image_size)
    else:
        test_dataset = build_test_dataset(data_dir, image_size=image_size)

    if test_dataset is None:
        return None, None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = getattr(test_dataset, "classes", None)
    if class_names is None:
        class_names = getattr(test_dataset, "class_names")

    return test_loader, class_names


def get_class_counts(dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    else:
        targets = [label for _, label in dataset.samples]

    counts = torch.bincount(torch.tensor(targets, dtype=torch.long))
    return counts


def build_dataloaders(
    data_dir: str | Path,
    dataset_type: str = "imagefolder",
    image_size: int = IMAGE_SIZE,
    batch_size: int = 32,
    num_workers: int = 4,
    backbone_type: str = "swin",
):
    if backbone_type == "vggface":
        train_t = build_vggface_train_transforms(image_size)
        val_t = build_vggface_eval_transforms(image_size)
    else:
        train_t = build_train_transforms(image_size)
        val_t = build_eval_transforms(image_size)

    if backbone_type == "vggface":
        data_dir = Path(data_dir)
        if dataset_type == "morph":
            train_dataset, val_dataset = build_morph_datasets(data_dir, image_size=image_size)
            train_dataset.transform = train_t
            val_dataset.transform = val_t
        elif dataset_type == "manifest":
            train_dataset, val_dataset = build_manifest_datasets(data_dir, image_size=image_size)
            train_dataset.transform = train_t
            val_dataset.transform = val_t
        else:
            data_dir = Path(data_dir)
            train_dataset = OrderedImageFolder(data_dir / "train", transform=train_t)
            val_dataset = OrderedImageFolder(data_dir / "val", transform=val_t)
    elif dataset_type == "morph":
        train_dataset, val_dataset = build_morph_datasets(data_dir, image_size=image_size)
    elif dataset_type == "manifest":
        train_dataset, val_dataset = build_manifest_datasets(data_dir, image_size=image_size)
    else:
        train_dataset, val_dataset = build_datasets(data_dir, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = getattr(train_dataset, "classes", None)
    if class_names is None:
        class_names = getattr(train_dataset, "class_names")

    return train_loader, val_loader, class_names
