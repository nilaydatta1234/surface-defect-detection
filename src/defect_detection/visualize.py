"""Side-by-side grid: ground truth vs each model's detections on the same test images."""

import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import CLASSES

SCALE = 2  # NEU-DET images are 200x200; upscale so boxes and labels are readable
GT_COLOR = (40, 200, 90)
PRED_COLOR = (240, 70, 60)


def draw(image: Image.Image, boxes, labels, color, scores=None, title: str = "") -> Image.Image:
    canvas = image.convert("RGB").resize(
        (image.width * SCALE, image.height * SCALE), Image.Resampling.NEAREST
    )
    pen = ImageDraw.Draw(canvas)
    for i, (box, label) in enumerate(zip(boxes.tolist(), labels.tolist(), strict=True)):
        x1, y1, x2, y2 = (v * SCALE for v in box)
        pen.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = CLASSES[label] + (f" {scores[i]:.2f}" if scores is not None else "")
        pen.text((x1 + 3, max(0, y1 - 12)), text, fill=color)
    if title:
        pen.rectangle((0, 0, canvas.width, 14), fill=(0, 0, 0))
        pen.text((4, 1), title, fill=(255, 255, 255))
    return canvas


def save_comparison_grid(
    images: list[Image.Image],
    targets: list[dict[str, torch.Tensor]],
    predictions: dict[str, list[dict[str, torch.Tensor]]],
    path: Path,
    per_class: int = 1,
    min_score: float = 0.3,
    seed: int = 0,
) -> None:
    """One row per sampled image: ground truth, then one column per model."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        if len(t["labels"]):
            by_class.setdefault(int(t["labels"][0]), []).append(i)
    picks = [
        i
        for c in sorted(by_class)
        for i in rng.sample(by_class[c], min(per_class, len(by_class[c])))
    ]

    rows = []
    for i in picks:
        tiles = [
            draw(
                images[i], targets[i]["boxes"], targets[i]["labels"], GT_COLOR, title="ground truth"
            )
        ]
        for name, preds in predictions.items():
            p = preds[i]
            keep = p["scores"] >= min_score
            tiles.append(
                draw(
                    images[i],
                    p["boxes"][keep],
                    p["labels"][keep],
                    PRED_COLOR,
                    p["scores"][keep],
                    name,
                )
            )
        rows.append(tiles)

    if not rows:
        return
    w, h = rows[0][0].size
    grid = Image.new("RGB", (w * len(rows[0]), h * len(rows)), "white")
    for r, tiles in enumerate(rows):
        for c, tile in enumerate(tiles):
            grid.paste(tile, (c * w, r * h))
    grid.save(path, quality=90)
