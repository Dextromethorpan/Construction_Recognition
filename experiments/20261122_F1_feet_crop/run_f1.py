import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
H = Path.home(); C = H / "GenIA-code"; E = Path(__file__).parent; V = H / "GenIA-data/labels/relations/v2x"
sys.path.insert(0, str(C / "08_controlled")); import judge
def feet(im, b):
    W, Hh = im.size; x, y, w, h = b; cx = (x + w / 2) / 100 * W
    x1, x2 = max(0, cx - w / 100 * W), min(W, cx + w / 100 * W); y1, y2 = max(0, (y + 0.7 * h) / 100 * Hh), min(Hh, (y + 1.3 * h) / 100 * Hh)
    c = im.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2))); s = 224 / max(c.size)
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s)))); cv = Image.new("RGB", (224, 224), (114, 114, 114))
    cv.paste(c, ((224 - c.size[0]) // 2, (224 - c.size[1]) // 2)); return cv
cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg); dets = judge.detections(cfg, imgs)
F, keys = judge.features(cfg, imgs, dets); row = {k: i for i, k in enumerate(keys)}; enc = judge.Encoder()
fc = E / "feet_train.npy"
if fc.exists(): FF = np.load(fc)
else:
    FF = np.zeros((len(keys), 256), np.float32); by = {}
    for n, k in enumerate(keys): by.setdefault(k[0], []).append((n, k))
    for g, lst in by.items():
        im = Image.open(imgs[g]["img"]).convert("RGB"); dp = [d for d in dets.get(g, []) if d["cls"] == 0]
        bx = [imgs[g]["people"][i]["box"] if kind == "gt" else tuple(dp[i]["box"]) for _, (_, kind, i) in lst]
        e, _ = enc([feet(im, b) for b in bx])
        for (n, _), v in zip(lst, e): FF[n] = v
    np.save(fc, FF)
XA = {"C1": F["C1"], "C1F": np.concatenate([F["C1"], FF], 1)}
# val_extra: D0 detections + features
from ultralytics import YOLO
P = list(csv.DictReader(open(V / "val_extra_people.csv"))); vimgs = sorted({p["gid"] for p in P}); by = {g: [p for p in P if p["gid"] == g] for g in vimgs}
m = YOLO(str(H / "GenIA-data/runs/ablation_v2b/V2_harness_field/weights/best.pt")); VX = {}
for g in vimgs:
    path = by[g][0]["img"]; r = m.predict(path, imgsz=512, conf=0.25, device="mps", verbose=False, max_det=300)[0]; W, Hh = r.orig_shape[1], r.orig_shape[0]
    dd = [(x1 / W * 100, y1 / Hh * 100, (x2 - x1) / W * 100, (y2 - y1) / Hh * 100) for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(c) == 0]
    if dd:
        im = Image.open(path).convert("RGB"); e1, _ = enc([judge.crop_c1(im, b) for b in dd]); ef, _ = enc([feet(im, b) for b in dd])
        x1 = np.array([np.concatenate([e1[j], judge.geo6(b, len(dd))]) for j, b in enumerate(dd)], np.float32)
        VX[g] = dict(d=dd, C1=x1, C1F=np.concatenate([x1, ef], 1))
    else: VX[g] = dict(d=[], C1=None, C1F=None)
sel = sorted(g for g, t in imgs.items() if t["set"] in judge.SEL_SETS); f = judge.TARGETS["on"]; cols = ["image_id", "person_id", "set", "on_scaffold", "score", "group_id", "matched"]
for arm in ("C1", "C1F"):
    X = XA[arm]
    for d in range(5):
        trg = judge.train_draw(imgs, d, 0.8); rr = [(row[(g, "gt", i)], f(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f(p) is not None]
        mdl = judge.fit(X[[r for r, _ in rr]], np.array([y for _, y in rr]), 1e-3); out = []
        for g in sel:
            t = imgs[g]; dp = [x for x in dets.get(g, []) if x["cls"] == 0]; mt = judge.match([p["box"] for p in t["people"]], [tuple(x["box"]) for x in dp], 0.5)
            s = mdl(X[[row[(g, "det", j)] for j in range(len(dp))]]) if dp else np.zeros(0)
            for i, p in enumerate(t["people"]):
                j = mt.get(i); out.append(dict(image_id=g, person_id=p["pid"], set=t["set"], on_scaffold=p["on"], group_id=t["group"], score=float(s[j]) if j is not None else 0.0, matched=int(j is not None)))
        with open(E / f"scores_{arm}_d{d}.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
        out = []
        for g in vimgs:
            v = VX[g]; mt = judge.match([tuple(float(p[k]) for k in "xywh") for p in by[g]], v["d"], 0.5); s = mdl(v[arm]) if v["d"] else np.zeros(0)
            for i, p in enumerate(by[g]):
                j = mt.get(i); out.append(dict(image_id=g, person_id=p["pid"], set="val_extra", on_scaffold=p["on"], group_id=g, score=float(s[j]) if j is not None else 0.0, matched=int(j is not None)))
        with open(E / f"xscores_{arm}_d{d}.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
    print(arm, "done", flush=True)
