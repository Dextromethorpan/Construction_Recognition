"""V3_confirm: parse val_extra labels, score D0 vs S (D1_960 conf0.10 NMS0.8) end-to-end with the C1 judge (5 draws),
and helmet C0 vs C0+head on labeled boxes. Selection-style threshold on val_extra not_on people. No test data."""
import csv, json, sys, subprocess
from pathlib import Path
import numpy as np, torch
from PIL import Image
H = Path.home(); C = H / "GenIA-code"; E = Path(__file__).parent; V = H / "GenIA-data/labels/relations/v2x"
sys.path.insert(0, str(C / "08_controlled")); import judge, threshold as T
sys.path.insert(0, str(C / "experiments/20261122_HL1_helmet_crop"))
order = [l.strip() for l in open(V / "val_extra_order.txt") if l.strip()]; P = json.load(open(V / "proposals.json"))
SUP = {"O": "on", "G": "not_on", "U": "unclear"}; HEL = {"W": "worn", "N": "not_worn", "H": "held", "V": "not_visible"}
people = []
for line in open(C / "02_labeling/val_extra/labels.txt"):
    k, rest = line.split("|", 1); k = int(k); g = order[k]; rest = rest.strip()
    if rest == "skip": continue
    n = 0
    for it in rest.split(";"):
        t = it.split()
        if not t: continue
        if t[0] == "add": box = tuple(float(v) for v in t[1].split(",")); s, h = t[2], t[3]
        else:
            if t[1] == "X": continue
            box = tuple(P[g]["boxes"][int(t[0])][:4]); s, h = t[1], t[2]
        people.append(dict(gid=g, k=k, pid=f"{g}#{n}", box=box, on=SUP[s], helmet=HEL[h], img=P[g]["img"])); n += 1
with open(V / "val_extra_people.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["gid", "k", "pid", "x", "y", "w", "h", "on", "helmet", "img"]); w.writeheader()
    for p in people: w.writerow({**{k: p[k] for k in ("gid", "k", "pid", "on", "helmet", "img")}, **dict(zip("xywh", p["box"]))})
imgs = sorted({p["gid"] for p in people}); by = {g: [p for p in people if p["gid"] == g] for g in imgs}
print("images", len(imgs), "people", len(people), {v: sum(p["on"] == v for p in people) for v in ("on", "not_on", "unclear")},
      {v: sum(p["helmet"] == v for p in people) for v in HEL.values()}, flush=True)

# --- train side (same as studies)
cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); timgs = judge.load_people(cfg)
F, keys = judge.features(cfg, timgs, judge.detections(cfg, timgs)); row = {k: i for i, k in enumerate(keys)}
z = np.load(C / "experiments/20261122_HL1_helmet_crop/head_features.npz", allow_pickle=True); HK = {tuple(k): i for i, k in enumerate(z["keys"])}; HF = z["F"]
def head_crop(im, b):
    W, Hh = im.size; x, y, w, h = b; cx = (x + w / 2) / 100 * W
    x1, x2 = max(0, cx - 0.5 * w / 100 * W), min(W, cx + 0.5 * w / 100 * W); y1, y2 = max(0, (y - 0.15 * h) / 100 * Hh), min(Hh, (y + 0.35 * h) / 100 * Hh)
    c = im.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2))); s = 224 / max(c.size)
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s)))); cv = Image.new("RGB", (224, 224), (114, 114, 114))
    cv.paste(c, ((224 - c.size[0]) // 2, (224 - c.size[1]) // 2)); return cv
heads = []
for d in range(5):
    trg = judge.train_draw(timgs, d, 0.8); m = {}
    rr = [(row[(g, "gt", i)], judge.TARGETS["on"](p)) for g in trg for i, p in enumerate(timgs[g]["people"]) if judge.TARGETS["on"](p) is not None]
    m["on"] = judge.fit(F["C1"][[r for r, _ in rr]], np.array([y for _, y in rr]), 1e-3)
    hr = [((g, "gt", i), judge.TARGETS["helmet"](p)) for g in trg for i, p in enumerate(timgs[g]["people"]) if judge.TARGETS["helmet"](p) is not None]
    X0 = F["C0"][[row[k] for k, _ in hr]]; XH = HF[[HK[k] for k, _ in hr]]; yh = np.array([y for _, y in hr])
    m["h0"] = judge.fit(X0, yh, 1e-3); m["h2"] = judge.fit(np.concatenate([X0, XH[:, :-6]], 1), yh, 1e-3); heads.append(m)

# --- detections for both arms
from ultralytics import YOLO
R = H / "GenIA-data/runs"
ARMS = {"D0": (str(R / "ablation_v2b/V2_harness_field/weights/best.pt"), 512, 0.25, 0.7),
        "S": (str(R / "D1_detector/D1_ft960/weights/best.pt"), 960, 0.10, 0.8)}
enc = judge.Encoder(); cache = {}
for g in imgs:
    im = Image.open(by[g][0]["img"]).convert("RGB"); gt = [p["box"] for p in by[g]]
    e0, _ = enc([judge.crop_c0(im, b) for b in gt]); eh, _ = enc([head_crop(im, b) for b in gt])
    cache[g] = dict(X0=np.array([np.concatenate([e0[i], judge.geo6(b, len(gt))]) for i, b in enumerate(gt)], np.float32), XH=np.array(eh, np.float32))
    for a, (w, isz, cf, iou) in ARMS.items():
        if a not in ARMS: continue
for a, (w, isz, cf, iou) in ARMS.items():
    m = YOLO(w); rows = {d: [] for d in range(5)}; spur = {d: [] for d in range(5)}
    for g in imgs:
        r = m.predict(by[g][0]["img"], imgsz=isz, conf=cf, iou=iou, device="mps", verbose=False, max_det=300)[0]; W, Hh = r.orig_shape[1], r.orig_shape[0]
        dets = [(x1 / W * 100, y1 / Hh * 100, (x2 - x1) / W * 100, (y2 - y1) / Hh * 100) for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(c) == 0]
        gt = [p["box"] for p in by[g]]; mt = judge.match(gt, dets, 0.5)
        if dets:
            im = Image.open(by[g][0]["img"]).convert("RGB"); e1, _ = enc([judge.crop_c1(im, b) for b in dets])
            X1 = np.array([np.concatenate([e1[j], judge.geo6(b, len(dets))]) for j, b in enumerate(dets)], np.float32)
        for d in range(5):
            s = heads[d]["on"](X1) if dets else np.zeros(0)
            for i, p in enumerate(by[g]):
                j = mt.get(i); rows[d].append(dict(image_id=g, person_id=p["pid"], set="val_extra", on_scaffold=p["on"], group_id=g,
                                                   score=round(float(s[j]), 6) if j is not None else 0.0, matched=int(j is not None)))
            spur[d] += [float(s[j]) for j in range(len(dets)) if j not in mt.values() and all(judge.iou(dets[j], b) < 0.3 for b in gt)]
    for d in range(5):
        with open(E / f"scores_{a}_d{d}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[d][0])); wr.writeheader(); wr.writerows(rows[d])
        json.dump(spur[d], open(E / f"spurious_{a}_d{d}.json", "w"))
    print(a, "done", flush=True)

# --- helmet
yl, p0, p2, grp = [], [[] for _ in range(5)], [[] for _ in range(5)], []
for g in imgs:
    for i, p in enumerate(by[g]):
        y = {"worn": 1, "not_worn": 0}.get(p["helmet"])
        if y is None: continue
        yl.append(y); grp.append(g); x0 = cache[g]["X0"][[i]]; xh = cache[g]["XH"][[i]]
        for d in range(5):
            p0[d].append(heads[d]["h0"](x0)[0]); p2[d].append(heads[d]["h2"](np.concatenate([x0, xh], 1))[0])
y = np.array(yl); grp = np.array(grp)
ba = lambda yy, pp: judge.bal_acc(yy, (np.asarray(pp) >= 0.5).astype(int))
A = [ba(y, p2[d]) for d in range(5)]; B = [ba(y, p0[d]) for d in range(5)]
ug = np.unique(grp); mem = {u: np.where(grp == u)[0] for u in ug}; rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    ix = np.concatenate([mem[u] for u in rng.choice(ug, len(ug))]); diffs.append(np.mean([ba(y[ix], np.asarray(p2[d])[ix]) - ba(y[ix], np.asarray(p0[d])[ix]) for d in range(5)]))
lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
nw = lambda pp: np.mean(np.asarray(pp)[y == 0] < 0.5)
print(f"HELMET n {len(y)} (worn {int(y.sum())}, not_worn {int((1 - y).sum())}): C0 {np.mean(B):.3f} vs C0+head {np.mean(A):.3f} | diff {np.mean(A) - np.mean(B):+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] | not_worn caught C0 {np.mean([nw(p0[d]) for d in range(5)]):.3f} vs {np.mean([nw(p2[d]) for d in range(5)]):.3f}")
