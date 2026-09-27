"""PyTorch dataset over the YOLO-format folders, producing torchvision detection targets."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, pil_to_tensor, vflip

from .data import find_images, read_yolo_labels


class DefectDataset(Dataset):
    """Returns (image, target) where target labels are class_id + 1 (0 = background)."""

    def __init__(self, root: Path, split: str, augment: bool = False) -> None:
        self.image_paths = sorted(find_images(Path(root) / "images" / split).values())
        self.label_dir = Path(root) / "labels" / split
        self.augment = augment
        if not self.image_paths:
            raise FileNotFoundError(f"no images in {Path(root) / 'images' / split}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def load(self, index: int) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        boxes = read_yolo_labels(self.label_dir / f"{path.stem}.txt", *image.size)
        xyxy = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32)
        labels = torch.tensor([b.label for b in boxes], dtype=torch.int64)
        return image, xyxy.reshape(-1, 4), labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, boxes, labels = self.load(index)
        tensor = pil_to_tensor(image).float() / 255.0
        _, height, width = tensor.shape

        if self.augment:
            # Surface defects have no canonical orientation, so both flips are label-safe
            if random.random() < 0.5:
                tensor = hflip(tensor)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            if random.random() < 0.5:
                tensor = vflip(tensor)
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

        return tensor, {"boxes": boxes, "labels": labels + 1}


def collate(batch):
    return tuple(zip(*batch, strict=True))
