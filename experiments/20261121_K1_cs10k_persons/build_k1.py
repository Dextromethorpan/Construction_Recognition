import csv, os, random, json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
D = Path.home() / "GenIA-data"; OUT = D / "build/K1_cs10k_persons"; SRC = D / "build/ablation_v2/V2_harness_field"
ph = {r["global_id"]: int(r["phash"], 16) for r in csv.DictReader(open(D / "reports/phash.csv"))}
held = {l.strip() for f in ("test_v1.txt", "cs10k_dev.txt") for l in open(D / "splits/v2" / f) if l.strip()}
cl = {r["global_id"]: r["cluster_id"] for r in csv.DictReader(open(D / "reports/clusters.csv"))}
bad_cl = {cl[g] for g in held if g in cl}
H = np.array([ph[g] for g in held if g in ph], dtype=np.uint64)
def mind(x):
    v = np.bitwise_xor(H, np.uint64(x)); return min(bin(int(t)).count("1") for t in v)
used = {l.strip() for f in ("train_v2.txt", "val_v2.txt") for l in open(D / "splits/v2" / f) if l.strip()}
pool = sorted(g for g in ph if g.startswith("cs10k/") and g not in held and g not in used and cl.get(g) not in bad_cl)
random.Random(0).shuffle(pool); pick = []
for g in pool:
    if mind(ph[g]) > 10: pick.append(g)
    if len(pick) == 2000: break
print("pool", len(pool), "picked", len(pick))
for s in ("train", "val"):
    (OUT / s / "images").mkdir(parents=True, exist_ok=True); (OUT / s / "labels").mkdir(parents=True, exist_ok=True)
    for p in (SRC / s / "images").iterdir():
        dst = OUT / s / "images" / p.name
        if not dst.exists(): os.symlink(p.resolve(), dst)
        lf = SRC / s / "labels" / (p.stem + ".txt")
        if lf.exists() and not (OUT / s / "labels" / lf.name).exists(): os.symlink(lf.resolve(), OUT / s / "labels" / lf.name)
m = YOLO(str(D / "runs/yolo11m.pt")); n_box = 0; log = {}
todo = [g for g in pick if not (OUT / "train/labels" / ("cs10k_" + g.split("/")[1] + ".txt")).exists()]
print("todo", len(todo), flush=True)
for k in range(0, len(todo), 4):
    b = todo[k:k + 4]; paths = [str(D / "extracted/cs10k" / (g.split("/")[1] + ".jpg")) for g in b]
    for g, p, r in zip(b, paths, m.predict(paths, imgsz=1280, conf=0.5, classes=[0], device="mps", verbose=False)):
        name = "cs10k_" + g.split("/")[1]; os.symlink(p, OUT / "train/images" / (name + ".jpg")) if not (OUT / "train/images" / (name + ".jpg")).exists() else None
        L = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in r.boxes.xywhn.tolist()]
        (OUT / "train/labels" / (name + ".txt")).write_text("\n".join(L) + ("\n" if L else "")); n_box += len(L); log[g] = len(L)
log = {g: len([l for l in open(OUT / "train/labels" / ("cs10k_" + g.split("/")[1] + ".txt")) if l.strip()]) for g in pick}; n_box = sum(log.values())
print("pseudo person boxes", n_box, "images with >=1", sum(v > 0 for v in log.values()))
json.dump({"picked": pick, "boxes": log}, open(OUT / "cs10k_pseudo.json", "w"))
(OUT / "data.yaml").write_text((SRC / "data.yaml").read_text().replace(str(SRC), str(OUT)))
