"""Builds a tiny synthetic dataset laid out like the Kaggle NEU-DET download."""

import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from defect_detection.data import CLASSES

VOC_TEMPLATE = """<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  {objects}
</annotation>"""

OBJECT_TEMPLATE = """<object><name>{name}</name><bndbox>
  <xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>
</bndbox></object>"""


def write_sample(raw: Path, name: str, index: int, rng: random.Random) -> None:
    stem = f"{name}_{index}"
    image = Image.new("L", (200, 200), color=128)
    pen = ImageDraw.Draw(image)
    objects = []
    for _ in range(rng.randint(1, 3)):
        x1, y1 = rng.randint(0, 150), rng.randint(0, 150)
        x2, y2 = x1 + rng.randint(15, 49), y1 + rng.randint(15, 49)
        pen.rectangle((x1, y1, x2, y2), fill=rng.randint(0, 60))
        objects.append(OBJECT_TEMPLATE.format(name=name, x1=x1, y1=y1, x2=x2, y2=y2))

    # Kaggle layout: images grouped in class folders, annotations flat
    (raw / "train" / "images" / name).mkdir(parents=True, exist_ok=True)
    (raw / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    image.save(raw / "train" / "images" / name / f"{stem}.jpg")
    (raw / "train" / "annotations" / f"{stem}.xml").write_text(
        VOC_TEMPLATE.format(stem=stem, objects="\n".join(objects))
    )


@pytest.fixture(scope="session")
def raw_dataset(tmp_path_factory) -> Path:
    raw = tmp_path_factory.mktemp("NEU-DET")
    rng = random.Random(0)
    for name in CLASSES:
        for i in range(1, 11):
            write_sample(raw, name, i, rng)
    return raw


@pytest.fixture(scope="session")
def prepared(raw_dataset, tmp_path_factory) -> Path:
    from defect_detection.prepare import prepare

    out = tmp_path_factory.mktemp("prepared") / "neu-det"
    prepare(raw_dataset, out)
    return out
