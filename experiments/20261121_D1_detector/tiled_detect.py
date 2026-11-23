"""Tiled inference: tiled_detect.py WEIGHTS IMGSZ OUT.json [--grid 2] [--overlap 0.2] [--sets ...]
Full image + grid x grid tiles (overlap), each at IMGSZ, conf 0.25 (fixed recipe); boxes merged per class with NMS 0.5.
Covers every pass-1 image judge.load_people() knows (same image list as judge.detections). Output format = judge cache."""
import argparse, json, sys, time
from pathlib import Path
import torch, torchvision
sys.path.insert(0, str(Path.home() / "GenIA-code/08_controlled"))
import judge
from PIL import Image
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("imgsz", type=int); ap.add_argument("out")
ap.add_argument("--grid", type=int, default=2); ap.add_argument("--overlap", type=float, default=0.2)
ap.add_argument("--config", default=str(Path.home() / "GenIA-code/experiments/20261121_D1_detector/arm_configs/D0.yaml"))
a = ap.parse_args()
cfg = judge.load_cfg(a.config); imgs = judge.load_people(cfg); m = YOLO(a.weights); out = {}; t0 = time.time()
for k, g in enumerate(sorted(imgs)):
    im = Image.open(imgs[g]["img"]).convert("RGB"); W, H = im.size
    crops = [(0, 0, W, H)]; n = a.grid; tw, th = W / (n - (n - 1) * a.overlap), H / (n - (n - 1) * a.overlap)
    for i in range(n):
        for j in range(n):
            x1, y1 = i * tw * (1 - a.overlap), j * th * (1 - a.overlap); crops.append((x1, y1, min(W, x1 + tw), min(H, y1 + th)))
    B, S, Cc = [], [], []
    for (x1, y1, x2, y2) in crops:
        r = m.predict(im.crop((round(x1), round(y1), round(x2), round(y2))), imgsz=a.imgsz, conf=0.25, device="mps", verbose=False, max_det=300)[0]
        if len(r.boxes):
            b = r.boxes.xyxy.cpu().clone(); b[:, [0, 2]] += round(x1); b[:, [1, 3]] += round(y1)
            B.append(b); S.append(r.boxes.conf.cpu()); Cc.append(r.boxes.cls.cpu())
    if B:
        b, s, c = torch.cat(B), torch.cat(S), torch.cat(Cc); keep = torchvision.ops.batched_nms(b, s, c.long(), 0.5)
        b, s, c = b[keep], s[keep], c[keep]
    else:
        b, s, c = torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)
    out[g] = [{"cls": int(ci), "conf": round(float(si), 4), "box": [x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100]}
              for (x1, y1, x2, y2), ci, si in zip(b.tolist(), c.tolist(), s.tolist())]
    if k % 200 == 0: print("tiled", k, len(imgs), f"{time.time() - t0:.0f}s", flush=True)
meta = {"_meta": {"weights": a.weights, "conf": 0.25, "imgsz": a.imgsz, "tiling": f"full + {a.grid}x{a.grid} overlap {a.overlap}, NMS 0.5",
                  "sec_per_image": round((time.time() - t0) / len(imgs), 4)}}
json.dump({**meta, **out}, open(a.out, "w")); print("done", meta)
