import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "GenIA-code/08_controlled")); import judge
from ultralytics import YOLO
W, IMGSZ, CONF, IOU, OUT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
cfg = judge.load_cfg(Path.home() / "GenIA-code/experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
m = YOLO(W); out = {"_meta": {"weights": W, "imgsz": IMGSZ, "conf": CONF, "iou": IOU}}
for g in sorted(imgs):
    r = m.predict(imgs[g]["img"], imgsz=IMGSZ, conf=CONF, iou=IOU, device="mps", verbose=False, max_det=300)[0]; Wd, H = r.orig_shape[1], r.orig_shape[0]
    out[g] = [{"cls": int(c), "conf": round(float(s), 4), "box": [x1 / Wd * 100, y1 / H * 100, (x2 - x1) / Wd * 100, (y2 - y1) / H * 100]}
              for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
json.dump(out, open(OUT, "w")); print("done", OUT)
