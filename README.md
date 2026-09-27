# surface-defect-detection

Detecting surface defects on hot-rolled steel with **YOLOv5** and **Faster R-CNN**, trained on the public [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) dataset and compared **on the same test split with the same evaluator**.

This is a public, reproducible version of work I did in industry, where I detected physical damage on LED display systems during production inspection. That work used proprietary images, so this repo uses NEU-DET instead.

<p align="center"><a href="https://colab.research.google.com/github/nilaydatta1234/surface-defect-detection/blob/main/notebooks/train_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a></p>

## Dataset

NEU-DET has 1,800 grayscale 200×200 images, 300 for each of six defect types: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale` and `scratches`. Annotations are Pascal VOC bounding boxes.

`defect_detection.prepare` converts the VOC XML to YOLO labels and makes a **stratified 70 / 15 / 15 split** (seed 42), so every class has the same share in train, val and test. The split is saved to `split_manifest.csv`, so results can be reproduced exactly. Both models train on the same YOLO-format files. The Faster R-CNN dataset converts the labels back to pixel boxes, so neither model gets different annotations.

## Models

| | YOLOv5s | Faster R-CNN |
|---|---|---|
| Type | one-stage, anchor-based | two-stage, region proposals |
| Backbone | CSPDarknet53 | ResNet-50 FPN (torchvision v2) |
| Init | COCO-pretrained | COCO-pretrained |
| Input | 640 px | 640 px (upscaled from 200 px, same as YOLO) |
| Schedule | 100 epochs, SGD, YOLOv5 default augmentation | 24 epochs, SGD + warm-up + cosine, horizontal/vertical flips, AMP |

## Evaluation

YOLOv5's `val.py` and torchvision's reference scripts compute mAP slightly differently (interpolation, detection caps), so comparing the numbers each one prints is misleading. Instead, `defect_detection.evaluate` collects raw predictions from both models (confidence ≥ 0.001, NMS IoU 0.6, max 100 detections per image) and scores them with one COCO-style implementation (`torchmetrics`). It reports:

- mAP@0.5 and mAP@0.5:0.95
- recall@100
- per-class AP@0.5
- median single-image latency

## Results

> Results will be filled in from the Colab run (`notebooks/train_colab.ipynb`). The notebook writes `results/metrics.json`, `results/results.md`, the training curves and a side-by-side prediction grid (`results/samples.jpg`).

## Reproduce

**Colab (recommended, about 1.5 h on a T4):** open the notebook with the badge above and run all cells. It asks for your `kaggle.json` to download the dataset.

**Locally:**

```bash
pip install -e ".[dev]"                       # plus a CUDA build of torch for training
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p raw --unzip
python -m defect_detection.prepare --raw raw --out data/neu-det

# YOLOv5
git clone https://github.com/ultralytics/yolov5 && pip install -r yolov5/requirements.txt
cd yolov5 && python train.py --img 640 --batch 32 --epochs 100 --seed 0 \
    --data ../data/neu-det/neu-det.yaml --weights yolov5s.pt --project ../runs/yolo --name neu-det && cd ..

# Faster R-CNN
python -m defect_detection.frcnn --data data/neu-det --epochs 24 --out runs/frcnn

# Score both on the test split
python -m defect_detection.evaluate --data data/neu-det --split test \
    --yolo runs/yolo/neu-det/weights/best.pt --yolo-repo yolov5 \
    --frcnn runs/frcnn/best.pt --out results
```

## Tests

```bash
pytest          # builds a synthetic NEU-DET-style dataset; includes a 1-step Faster R-CNN training run on CPU
ruff check .
```

## Layout

```
src/defect_detection/
  data.py        VOC parsing, YOLO label conversion, stratified split
  prepare.py     raw NEU-DET -> YOLO dataset + manifest
  dataset.py     PyTorch dataset with label-safe flip augmentation
  frcnn.py       Faster R-CNN fine-tuning
  metrics.py     shared COCO-style evaluator
  evaluate.py    test-set comparison, latency, report
  visualize.py   ground truth vs predictions grid
notebooks/train_colab.ipynb
```

## License

MIT. NEU-DET is provided by Northeastern University (China) for research use.
