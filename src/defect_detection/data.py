"""Dataset handling for NEU-DET: VOC XML parsing, YOLO label format, stratified splits.

YOLO txt labels are the single on-disk format. YOLOv5 trains on them directly and the
Faster R-CNN dataset converts them back to pixel boxes, so both models see identical
annotations and splits.
"""

import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Box:
    label: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Annotation:
    stem: str
    width: int
    height: int
    boxes: tuple[Box, ...]

    @property
    def primary_class(self) -> int:
        """NEU-DET images each show one defect type; used to stratify splits."""
        return self.boxes[0].label if self.boxes else -1


def parse_voc(xml_path: Path) -> Annotation:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    boxes = []
    for obj in root.iter("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"{xml_path.name}: unknown class '{name}'")
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # A few NEU-DET boxes touch or exceed the border; clip so they stay valid
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append(Box(CLASS_TO_ID[name], x1, y1, x2, y2))
    return Annotation(xml_path.stem, width, height, tuple(boxes))


def to_yolo_line(box: Box, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return f"{box.label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def from_yolo_line(line: str, width: int, height: int) -> Box:
    label, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
    return Box(int(label), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return [from_yolo_line(ln, width, height) for ln in lines]


def stratified_split(
    annotations: list[Annotation],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Annotation]]:
    """Split per class so every defect type has the same share in train/val/test."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")

    by_class: dict[int, list[Annotation]] = defaultdict(list)
    for ann in sorted(annotations, key=lambda a: a.stem):
        by_class[ann.primary_class].append(ann)

    rng = random.Random(seed)
    splits: dict[str, list[Annotation]] = {"train": [], "val": [], "test": []}
    for items in by_class.values():
        rng.shuffle(items)
        n_train = round(len(items) * fractions[0])
        n_val = round(len(items) * fractions[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train : n_train + n_val]
        splits["test"] += items[n_train + n_val :]
    return splits


def find_images(root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return {p.stem: p for p in root.rglob("*") if p.suffix.lower() in exts}
