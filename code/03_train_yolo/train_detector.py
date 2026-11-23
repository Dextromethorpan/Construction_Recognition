"""Train the scheme-B detector once (person, helmet, head, harness, scaffold_platform, scaffold_structure).
Usage: train_detector.py --model yolo11n.pt --imgsz 640 --epochs 30 --name det_B_v1"""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import argparse
from ultralytics import YOLO
p = argparse.ArgumentParser()
p.add_argument("--model", default="yolo11n.pt"); p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch", type=int, default=16)
p.add_argument("--name", default="det_B_v1"); p.add_argument("--data", default=GENIA_DATA + "/build/ablation_v1/base_data.yaml")
a = p.parse_args()
YOLO(a.model).train(data=a.data,
    epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device="mps", workers=4, seed=0, patience=8,
    cos_lr=True, close_mosaic=5, plots=True, project=GENIA_DATA + "/runs/detector",
    name=a.name, exist_ok=False)
