import random

import pytest
import torch

from defect_detection.dataset import DefectDataset
from defect_detection.evaluate import markdown_table
from defect_detection.metrics import detection_metrics
from defect_detection.visualize import save_comparison_grid


def test_dataset_targets_match_labels(prepared):
    ds = DefectDataset(prepared, "train")
    image, target = ds[0]
    assert image.shape == (3, 200, 200)
    assert image.max() <= 1.0
    assert target["labels"].min() >= 1  # 0 is reserved for background
    assert target["boxes"].shape[1] == 4


def test_flip_augmentation_keeps_boxes_valid(prepared):
    ds = DefectDataset(prepared, "train", augment=True)
    random.seed(1)
    for i in range(len(ds)):
        _, target = ds[i]
        boxes = target["boxes"]
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert boxes.min() >= 0 and boxes.max() <= 200


def test_perfect_predictions_score_one():
    targets = [
        {"boxes": torch.tensor([[10.0, 10, 50, 50]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[60.0, 60, 120, 140]]), "labels": torch.tensor([5])},
    ]
    preds = [{**t, "scores": torch.tensor([0.9])} for t in targets]
    m = detection_metrics(preds, targets)
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)
    assert m["ap50/crazing"] == pytest.approx(1.0)
    assert m["ap50/scratches"] == pytest.approx(1.0)


def test_wrong_class_scores_zero():
    targets = [{"boxes": torch.tensor([[10.0, 10, 50, 50]]), "labels": torch.tensor([0])}]
    preds = [
        {"boxes": targets[0]["boxes"], "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}
    ]
    assert detection_metrics(preds, targets)["map50"] == 0.0


def test_markdown_table_lists_every_model():
    metrics = {"map50": 0.5, "map": 0.25, "mar100": 0.6, "latency_ms": 12.3, "ap50/crazing": 0.4}
    table = markdown_table(
        {"A": metrics, "B": metrics, "_meta": {"images": 3, "device": "cpu", "evaluator": "x"}}
    )
    assert "| Metric | A | B |" in table
    assert "| AP@0.5 · crazing | 0.400 | 0.400 |" in table


def test_comparison_grid(prepared, tmp_path):
    ds = DefectDataset(prepared, "test")
    images, targets = [], []
    for i in range(len(ds)):
        image, boxes, labels = ds.load(i)
        images.append(image)
        targets.append({"boxes": boxes, "labels": labels})
    preds = [{**t, "scores": torch.ones(len(t["labels"]))} for t in targets]

    out = tmp_path / "grid.jpg"
    save_comparison_grid(images, targets, {"model": preds}, out)
    from PIL import Image

    assert Image.open(out).size == (2 * 400, 6 * 400)  # 2 columns, one row per class


@pytest.mark.slow
def test_frcnn_trains_one_step_and_evaluates(prepared, tmp_path):
    from defect_detection.frcnn import train

    metrics = train(
        prepared, tmp_path, epochs=1, batch_size=2, workers=0, max_batches=1, pretrained=False
    )
    assert (tmp_path / "best.pt").exists()
    assert 0.0 <= metrics["map50"] <= 1.0
