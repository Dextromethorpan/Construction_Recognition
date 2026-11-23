"""Detections for one exported variant in judge cache format: detect_variant.py MODEL IMGSZ DEVICE OUT.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "GenIA-code/08_controlled")); import judge
from ultralytics import YOLO
M, IMGSZ, DEV, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cfg = judge.load_cfg(Path.home() / "GenIA-code/experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
m = YOLO(M, task="detect"); out = {}; t0 = time.time()
for g in sorted(imgs):
    r = m.predict(imgs[g]["img"], imgsz=IMGSZ, conf=0.25, device=DEV, verbose=False, max_det=300)[0]
    W, H = r.orig_shape[1], r.orig_shape[0]
    out[g] = [{"cls": int(c), "conf": round(float(s), 4), "box": [x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100]}
              for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
json.dump({"_meta": {"weights": M, "conf": 0.25, "imgsz": IMGSZ, "device": DEV, "sec_per_image": round((time.time() - t0) / len(imgs), 4)}, **out}, open(OUT, "w"))
print("done", OUT)
