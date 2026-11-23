"""Step 1: complete the person labels. Many source datasets label only helmets or some people, so a
COCO-pretrained YOLO (yolo11m) adds person boxes (conf >= 0.5) that don't overlap an existing person box
(IoU < 0.5). Writes build/B_persons_v1 (same images as B via symlink, new label files, added boxes logged)."""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import os, json, shutil
from pathlib import Path
from ultralytics import YOLO

B = Path(GENIA_DATA + "/build/B_objects_platform_v1")
OUT = Path(GENIA_DATA + "/build/B_persons_v1")
CONF, IOU_MAX = 0.5, 0.5

def iou(a, b):  # cx,cy,w,h normalized
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw, ih = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    i = iw*ih; u = a[2]*a[3]+b[2]*b[3]-i
    return i/u if u else 0

m = YOLO("yolo11m.pt"); log = {}
for split in ("train", "val"):
    (OUT/split/"labels").mkdir(parents=True, exist_ok=True)
    # per-file symlinks: a symlinked images/ dir makes Ultralytics resolve to the OLD labels/ folder
    (OUT/split/"images").mkdir(exist_ok=True)
    for p in (B/split/"images").iterdir():
        if not (OUT/split/"images"/p.name).exists(): os.symlink(p.resolve(), OUT/split/"images"/p.name)
    imgs = sorted((B/split/"images").iterdir()); added = 0
    for k in range(0, len(imgs), 32):
        batch = imgs[k:k+32]
        for p, r in zip(batch, m.predict([str(x) for x in batch], imgsz=640, conf=CONF, classes=[0], device="mps", verbose=False)):
            lf = B/split/"labels"/f"{p.stem}.txt"
            lines = [l.strip() for l in open(lf) if l.strip()] if lf.exists() else []
            persons = [tuple(map(float, l.split()[1:])) for l in lines if l.split()[0] == "0"]
            new = [b for b in r.boxes.xywhn.tolist() if all(iou(b, q) < IOU_MAX for q in persons)]
            lines += [f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in new]
            (OUT/split/"labels"/f"{p.stem}.txt").write_text("\n".join(lines)+"\n")
            if new: log[f"{split}/{p.stem}"] = len(new); added += len(new)
        if k % 3200 == 0: print(split, k, len(imgs), added, flush=True)
    print(split, "added person boxes:", added, flush=True)
json.dump(log, open(OUT/"added_persons.json", "w"))
(OUT/"data.yaml").write_text((Path(GENIA_DATA + "/build/ablation_v1/base_data.yaml").read_text()
    .replace(str(B), str(OUT))))
