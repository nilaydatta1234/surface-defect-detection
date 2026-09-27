"""Fine-tune torchvision Faster R-CNN (ResNet-50 FPN v2, COCO-pretrained) on NEU-DET.

python -m defect_detection.frcnn --data data/neu-det --epochs 24 --out runs/frcnn
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .data import CLASSES
from .dataset import DefectDataset, collate
from .metrics import detection_metrics

# NEU-DET images are 200x200. YOLOv5 is trained at 640, so upscale to the same size here
# to keep the comparison about the architectures rather than the input resolution.
IMAGE_SIZE = 640


def build_model(pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        box_score_thresh=0.001,  # keep low-confidence boxes; mAP needs the full ranking
        box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model


def load_model(weights: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, images: list[torch.Tensor], device) -> list[dict[str, torch.Tensor]]:
    outputs = model([img.to(device) for img in images])
    return [
        {"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu() - 1}
        for o in outputs
    ]


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    preds, targets = [], []
    for images, batch_targets in loader:
        preds += predict(model, images, device)
        targets += [{"boxes": t["boxes"], "labels": t["labels"] - 1} for t in batch_targets]
    return detection_metrics(preds, targets)


def train(
    data: Path,
    out: Path,
    epochs: int = 24,
    batch_size: int = 8,
    lr: float = 0.01,
    workers: int = 2,
    max_batches: int | None = None,
    pretrained: bool = True,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    train_loader = DataLoader(
        DefectDataset(data, "train", augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        DefectDataset(data, "val"), batch_size=batch_size, num_workers=workers, collate_fn=collate
    )

    model = build_model(pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    steps_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = min(300, total_steps // 10)

    def lr_lambda(step: int) -> float:  # linear warmup, then cosine decay
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_map50, best_metrics = -1.0, {}
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "val_map50", "val_map", "seconds"])

    for epoch in range(1, epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for i, (images, targets) in enumerate(train_loader):
            if max_batches and i >= max_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        metrics = evaluate(model, val_loader, device)
        seconds = time.time() - start
        avg_loss = running / steps_per_epoch
        print(
            f"epoch {epoch}/{epochs}  loss {avg_loss:.4f}  "
            f"val mAP50 {metrics['map50']:.4f}  mAP50-95 {metrics['map']:.4f}  ({seconds:.0f}s)"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{avg_loss:.4f}",
                    f"{metrics['map50']:.4f}",
                    f"{metrics['map']:.4f}",
                    f"{seconds:.0f}",
                ]
            )
        torch.save(model.state_dict(), out / "last.pt")
        if metrics["map50"] > best_map50:
            best_map50, best_metrics = metrics["map50"], metrics
            torch.save(model.state_dict(), out / "best.pt")

    print(f"best val mAP50 {best_map50:.4f} -> {out / 'best.pt'}")
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/neu-det"))
    parser.add_argument("--out", type=Path, default=Path("runs/frcnn"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.workers)


if __name__ == "__main__":
    main()
