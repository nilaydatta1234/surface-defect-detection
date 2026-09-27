"""Score YOLOv5 and Faster R-CNN on the held-out test split with the same evaluator.

    python -m defect_detection.evaluate --data data/neu-det \
        --frcnn runs/frcnn/best.pt \
        --yolo runs/yolo/neu-det/weights/best.pt --yolo-repo yolov5 \
        --out results
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from .data import CLASSES
from .dataset import DefectDataset
from .metrics import detection_metrics
from .visualize import save_comparison_grid

CONF_FOR_MAP = 0.001  # mAP must see low-confidence detections too
NMS_IOU = 0.6


def load_yolo(weights: Path, repo: Path | None, device: torch.device):
    if repo:
        model = torch.hub.load(str(repo), "custom", path=str(weights), source="local")
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights))
    model.conf, model.iou, model.max_det = CONF_FOR_MAP, NMS_IOU, 100
    return model.to(device).eval()


@torch.no_grad()
def yolo_predict(model, images: list[Image.Image]) -> list[dict[str, torch.Tensor]]:
    results = model(images, size=640)
    return [
        {"boxes": det[:, :4].cpu(), "scores": det[:, 4].cpu(), "labels": det[:, 5].long().cpu()}
        for det in results.xyxy
    ]


@torch.no_grad()
def frcnn_predict(model, images: list[Image.Image], device) -> list[dict[str, torch.Tensor]]:
    from torchvision.transforms.functional import pil_to_tensor

    from .frcnn import predict

    tensors = [pil_to_tensor(img).float() / 255.0 for img in images]
    return predict(model, tensors, device)


def median_latency_ms(fn, images: list[Image.Image], device, n: int = 50) -> float:
    """Single-image latency (batch 1), median over n images after warm-up."""
    for img in images[:5]:
        fn([img])
    timings = []
    for img in images[:n]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn([img])
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def run(data: Path, split: str, frcnn: Path | None, yolo: Path | None, yolo_repo, out: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DefectDataset(data, split)
    images, targets = [], []
    for i in range(len(dataset)):
        image, boxes, labels = dataset.load(i)
        images.append(image)
        targets.append({"boxes": boxes, "labels": labels})

    predictors = {}
    if yolo:
        yolo_model = load_yolo(yolo, yolo_repo, device)
        predictors["YOLOv5s"] = lambda batch: yolo_predict(yolo_model, batch)
    if frcnn:
        from .frcnn import load_model

        frcnn_model = load_model(frcnn, device)
        predictors["Faster R-CNN R50-FPN v2"] = lambda batch: frcnn_predict(
            frcnn_model, batch, device
        )
    if not predictors:
        raise SystemExit("pass --yolo and/or --frcnn")

    out.mkdir(parents=True, exist_ok=True)
    report, all_preds = {}, {}
    for name, fn in predictors.items():
        preds = []
        for start in range(0, len(images), 16):
            preds += fn(images[start : start + 16])
        metrics = detection_metrics(preds, targets)
        metrics["latency_ms"] = median_latency_ms(fn, images, device)
        report[name] = metrics
        all_preds[name] = preds
        print(
            f"{name}: mAP50 {metrics['map50']:.3f}  mAP50-95 {metrics['map']:.3f}  "
            f"{metrics['latency_ms']:.1f} ms/img"
        )

    report["_meta"] = {
        "split": split,
        "images": len(images),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "evaluator": "torchmetrics MeanAveragePrecision (COCO-style)",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "results.md").write_text(markdown_table(report), encoding="utf-8")
    save_comparison_grid(images, targets, all_preds, out / "samples.jpg")
    print(f"wrote {out / 'metrics.json'}, {out / 'results.md'}, {out / 'samples.jpg'}")
    return report


def markdown_table(report: dict) -> str:
    models = [k for k in report if not k.startswith("_")]
    meta = report["_meta"]
    rows = [
        f"Test split: {meta['images']} images · device: {meta['device']} · {meta['evaluator']}",
        "",
        "| Metric | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]

    def row(label: str, key: str, fmt: str = "{:.3f}") -> None:
        rows.append(f"| {label} | " + " | ".join(fmt.format(report[m][key]) for m in models) + " |")

    row("mAP@0.5", "map50")
    row("mAP@0.5:0.95", "map")
    row("Recall@100", "mar100")
    row("Latency (ms/img, batch 1)", "latency_ms", "{:.1f}")
    for cls in CLASSES:
        if all(f"ap50/{cls}" in report[m] for m in models):
            row(f"AP@0.5 · {cls}", f"ap50/{cls}")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/neu-det"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--frcnn", type=Path)
    parser.add_argument("--yolo", type=Path)
    parser.add_argument("--yolo-repo", type=Path, help="local clone of ultralytics/yolov5")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data, args.split, args.frcnn, args.yolo, args.yolo_repo, args.out)


if __name__ == "__main__":
    main()
