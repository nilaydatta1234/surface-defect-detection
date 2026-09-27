import csv
from collections import Counter

import pytest
import yaml

from defect_detection.data import (
    CLASSES,
    Annotation,
    Box,
    from_yolo_line,
    parse_voc,
    stratified_split,
    to_yolo_line,
)


def test_yolo_roundtrip():
    box = Box(3, 10.0, 20.0, 110.0, 70.0)
    line = to_yolo_line(box, 200, 200)
    assert line == "3 0.300000 0.225000 0.500000 0.250000"
    back = from_yolo_line(line, 200, 200)
    assert back.label == 3
    assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx((10, 20, 110, 70))


def test_parse_voc_clips_and_drops_degenerate_boxes(tmp_path):
    xml = tmp_path / "patches_1.xml"
    xml.write_text(
        """<annotation><size><width>200</width><height>200</height></size>
        <object><name>patches</name><bndbox>
          <xmin>-5</xmin><ymin>10</ymin><xmax>250</xmax><ymax>60</ymax></bndbox></object>
        <object><name>patches</name><bndbox>
          <xmin>30</xmin><ymin>30</ymin><xmax>30</xmax><ymax>80</ymax></bndbox></object>
        </annotation>"""
    )
    ann = parse_voc(xml)
    assert ann.boxes == (Box(CLASSES.index("patches"), 0.0, 10.0, 200.0, 60.0),)


def test_parse_voc_rejects_unknown_class(tmp_path):
    xml = tmp_path / "x.xml"
    xml.write_text(
        "<annotation><size><width>200</width><height>200</height></size>"
        "<object><name>rust</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax>"
        "<ymax>9</ymax></bndbox></object></annotation>"
    )
    with pytest.raises(ValueError, match="rust"):
        parse_voc(xml)


def test_stratified_split_is_balanced_and_deterministic():
    anns = [
        Annotation(f"{c}_{i}", 200, 200, (Box(c, 0, 0, 10, 10),))
        for c in range(6)
        for i in range(300)
    ]
    first = stratified_split(anns, seed=42)
    second = stratified_split(list(reversed(anns)), seed=42)

    assert {k: [a.stem for a in v] for k, v in first.items()} == {
        k: [a.stem for a in v] for k, v in second.items()
    }
    for split, expected in (("train", 210), ("val", 45), ("test", 45)):
        per_class = Counter(a.primary_class for a in first[split])
        assert set(per_class.values()) == {expected}
    all_stems = [a.stem for items in first.values() for a in items]
    assert len(all_stems) == len(set(all_stems)) == 1800


def test_prepare_writes_yolo_dataset(prepared):
    config = yaml.safe_load((prepared / "neu-det.yaml").read_text())
    assert config["names"] == dict(enumerate(CLASSES))

    counts = {s: len(list((prepared / "images" / s).iterdir())) for s in ("train", "val", "test")}
    assert counts == {"train": 42, "val": 12, "test": 6}  # 10 per class: 7 / 2 / 1 after rounding

    for split in counts:
        images = {p.stem for p in (prepared / "images" / split).iterdir()}
        labels = {p.stem for p in (prepared / "labels" / split).iterdir()}
        assert images == labels

    with open(prepared / "split_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    assert {r["class"] for r in rows if r["split"] == "test"} == set(CLASSES)
