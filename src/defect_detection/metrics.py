"""One evaluator for both models, so the numbers are directly comparable.

YOLOv5's val.py and torchvision's reference scripts each compute mAP slightly
differently (interpolation, max detections), so both models' raw predictions are
scored here with the same COCO-style implementation instead.
"""

import torch
from torchmetrics.detection import MeanAveragePrecision

from .data import CLASSES


def detection_metrics(
    preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]
) -> dict[str, float]:
    """preds: boxes (xyxy px), scores, labels (0-based). targets: boxes, labels."""
    coco = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    coco.update(preds, targets)
    overall = coco.compute()

    at_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)
    at_50.update(preds, targets)
    per_class = at_50.compute()

    result = {
        "map50": float(overall["map_50"]),
        "map": float(overall["map"]),
        "mar100": float(overall["mar_100"]),
    }
    classes = per_class["classes"].tolist()
    ap50 = per_class["map_per_class"].tolist()
    if isinstance(classes, int):  # single class present returns scalars
        classes, ap50 = [classes], [ap50]
    for cls, ap in zip(classes, ap50, strict=True):
        result[f"ap50/{CLASSES[cls]}"] = float(ap)
    return result
