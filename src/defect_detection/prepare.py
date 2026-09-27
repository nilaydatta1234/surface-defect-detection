"""Convert raw NEU-DET (VOC XML) into a YOLO-format dataset with a fixed 70/15/15 split.

python -m defect_detection.prepare --raw /path/to/NEU-DET --out data/neu-det
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml

from .data import CLASSES, find_images, parse_voc, stratified_split, to_yolo_line


def prepare(raw: Path, out: Path, seed: int = 42) -> dict[str, int]:
    xml_files = sorted(raw.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No VOC .xml annotations found under {raw}")
    images = find_images(raw)

    annotations, skipped = [], []
    for xml_path in xml_files:
        ann = parse_voc(xml_path)
        if ann.stem not in images or not ann.boxes:
            skipped.append(ann.stem)
            continue
        annotations.append(ann)

    if out.exists():
        shutil.rmtree(out)
    splits = stratified_split(annotations, seed=seed)

    manifest = []
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for ann in items:
            src = images[ann.stem]
            shutil.copy2(src, out / "images" / split / src.name)
            lines = [to_yolo_line(b, ann.width, ann.height) for b in ann.boxes]
            (out / "labels" / split / f"{ann.stem}.txt").write_text("\n".join(lines) + "\n")
            manifest.append((ann.stem, split, CLASSES[ann.primary_class], len(ann.boxes)))

    with open(out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "class", "num_boxes"])
        writer.writerows(sorted(manifest))

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }
    (out / "neu-det.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    counts = {split: len(items) for split, items in splits.items()}
    print(f"Prepared {sum(counts.values())} images -> {out}  {counts}")
    if skipped:
        print(f"Skipped {len(skipped)} annotations without image or boxes: {skipped[:5]}...")
    boxes = Counter(CLASSES[b.label] for a in annotations for b in a.boxes)
    print("Boxes per class:", dict(sorted(boxes.items())))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="unzipped NEU-DET folder")
    parser.add_argument("--out", type=Path, default=Path("data/neu-det"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw, args.out, args.seed)


if __name__ == "__main__":
    main()
