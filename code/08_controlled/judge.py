#!/usr/bin/env python3
"""On-scaffold judge for controlled studies (decision_rule.md). Ideas from 07_ablation/design_B.py + combined.py (not edited).

Pipeline: fixed detector (cached) -> person boxes -> features of input C0..C3 (cached) -> class-balanced L2 logistic
regression (fixed recipe) -> probability. Labels: frozen_v2 pass 1, on_scaffold derived, unclear/unannotated dropped.

  judge.py prepare --config CFG              detections for every pass-1 image + features for all inputs (cached)
  judge.py run --config CFG --arm C0 --seed 0 --outdir DIR [--train-frac F]
      -> DIR/scores_<arm>_d<seed>.csv       end-to-end (val_v2 + cs10k_dev labeled people; detector miss = score 0)
      -> DIR/scores_cond_<arm>_d<seed>.csv  conditional (labeled boxes as input)
      -> DIR/extra_<arm>_d<seed>.json       conditional AP, helmet / harness balanced accuracy, counts, spurious detections
Score CSV columns follow threshold.py: image_id, person_id, set, on_scaffold, score, group_id (+ matched, det_conf).
Train draws: seed d picks a random `train_fraction_per_draw` of train_v2 images (grouped by image); same for every arm.
"""
import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

import os


def pick_device():
    """Torch device: GENIA_DEVICE if set, else Apple GPU (mps), else NVIDIA GPU (cuda), else CPU. On a Mac this is mps, as before."""
    if os.environ.get("GENIA_DEVICE"): return os.environ["GENIA_DEVICE"]
    if torch.backends.mps.is_available(): return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()
HOME = Path.home(); D = Path(os.environ.get("GENIA_DATA", HOME / "GenIA-data")); C = Path(os.environ.get("GENIA_CODE", HOME / "GenIA-code"))
FROZEN = D / "labels/relations/v2/frozen_v2"
EMB_W = D / "runs/yolo11n-cls.pt"
INPUTS = ("C0", "C1", "C2", "C3", "C0G", "C1G")   # C0G / C1G = + structure geometry (study C2_structure)
SEL_SETS = ("val_v2", "cs10k_dev")


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HOME / p


def load_cfg(path):
    from ultralytics.utils import YAML      # pyyaml is not a direct dep of .venv-train
    cfg = YAML.load(str(path)); cfg["_path"] = str(path)
    cache_dir = resolve(cfg["detector"]["cache"]).parent; cache_dir.mkdir(parents=True, exist_ok=True)
    cfg["_cache_dir"] = cache_dir
    return cfg


# ---------------------------------------------------------------- data
def load_people(cfg):
    """Pass-1 images and labeled people from frozen_v2. Returns {gid: {...}} with all people (incl. unclear)."""
    tasks = {t["data"]["global_id"]: t for t in json.load(open(FROZEN / "labels_v2.json")) if t["data"]["pass"] == 1}
    imgs = {}
    for r in csv.DictReader(open(FROZEN / "people_v2.csv")):
        if r["pass"] != "1": continue
        g = r["global_id"]
        if g not in imgs:
            d = tasks[g]["data"]
            imgs[g] = dict(gid=g, set=d["split_v2"], synthetic=int(d.get("synthetic") or 0),
                           img=str(D / d["image"].split("?d=")[1]), group=r["site_or_sequence_id"] or g, people=[])
        imgs[g]["people"].append(dict(pid=r["person_id"], box=tuple(float(r[k]) for k in "xywh"),
                                      on=r["on_scaffold"], support=r["support_surface"], helmet=r["helmet_state"],
                                      harness=r["harness_state"], review=r["review_status"], hn=r["hard_negative_type"]))
    for g, t in tasks.items():          # images with no labeled person still get detections (spurious FA count)
        if g not in imgs:
            d = t["data"]
            imgs[g] = dict(gid=g, set=d["split_v2"], synthetic=int(d.get("synthetic") or 0),
                           img=str(D / d["image"].split("?d=")[1]), group=g, people=[])
    if cfg["data"]["series"] == "real_only":
        imgs = {g: v for g, v in imgs.items() if not v["synthetic"]}
    return imgs


def iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = iw * ih; u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u else 0


def match(gt, dets, thr):
    """Greedy best-IoU-first. Returns {gt_index: det_index}."""
    pairs = sorted(((iou(g, d), i, j) for i, g in enumerate(gt) for j, d in enumerate(dets)), reverse=True)
    out, used = {}, set()
    for s, i, j in pairs:
        if s < thr: break
        if i in out or j in used: continue
        out[i] = j; used.add(j)
    return out


# ---------------------------------------------------------------- detections
def detections(cfg, imgs):
    path = resolve(cfg["detector"]["cache"])
    if path.exists(): return json.load(open(path))
    from ultralytics import YOLO
    dc = cfg["detector"]; m = YOLO(dc["weights"]); out = {}
    for k, g in enumerate(sorted(imgs)):
        r = m.predict(imgs[g]["img"], imgsz=dc["imgsz"], conf=dc["conf"], device=DEVICE, verbose=False, max_det=300)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        out[g] = [{"cls": int(c), "conf": round(float(s), 4), "box": [x1 / W * 100, y1 / H * 100, (x2 - x1) / W * 100, (y2 - y1) / H * 100]}
                  for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        if k % 200 == 0: print("detect", k, len(imgs), flush=True)
    meta = {"_meta": {"weights": dc["weights"], "conf": dc["conf"], "imgsz": dc["imgsz"]}}
    json.dump({**meta, **out}, open(path, "w"))
    return json.load(open(path))


# ---------------------------------------------------------------- features
class Encoder:
    """yolo11n-cls, frozen. Returns (256-d global-pooled backbone map = design_B embedding, same map pooled to 3x3)."""
    def __init__(self):
        from ultralytics import YOLO
        self.net = YOLO(str(EMB_W)).model.model.eval().to(DEVICE)

    @torch.no_grad()
    def __call__(self, crops):
        x = torch.stack([torch.from_numpy(np.asarray(c, dtype=np.float32) / 255.).permute(2, 0, 1) for c in crops]).to(DEVICE)
        for layer in self.net[:-1]: x = layer(x)
        emb = x.mean((2, 3))      # = ultralytics embed() for -cls models (global pool of the 256-ch map), as in design_B
        sp = torch.nn.functional.adaptive_avg_pool2d(x.cpu(), 3).flatten(1)   # MPS cannot pool 7x7 -> 3x3
        return emb.cpu().numpy(), sp.numpy()


def crop_c0(im, b):
    W, H = im.size; x, y, w, h = b; cx, cy = x + w / 2, y + h / 2
    x1, x2 = max(0, cx - w), min(100, cx + w); y1, y2 = max(0, cy - 0.75 * h), min(100, cy + 0.75 * h)
    return im.crop((x1 / 100 * W, y1 / 100 * H, max(x2 / 100 * W, x1 / 100 * W + 2), max(y2 / 100 * H, y1 / 100 * H + 2))).resize((224, 224))


def crop_c1(im, b):
    """3x width, 2x height around the box centre, clipped to the image, letterboxed to a square (aspect kept)."""
    W, H = im.size; x, y, w, h = b; cx, cy = (x + w / 2) / 100 * W, (y + h / 2) / 100 * H
    hw, hh = 1.5 * w / 100 * W, 1.0 * h / 100 * H
    x1, x2, y1, y2 = max(0, cx - hw), min(W, cx + hw), max(0, cy - hh), min(H, cy + hh)
    c = im.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2)))
    s = 224 / max(c.size); c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s))))
    canvas = Image.new("RGB", (224, 224), (114, 114, 114)); canvas.paste(c, ((224 - c.size[0]) // 2, (224 - c.size[1]) // 2))
    return canvas


def geo6(b, n):
    x, y, w, h = b
    return [(x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100, (y + h) / 100, n / 10]


def support_geo(b, plats):
    """7 numbers: foot point (box bottom-centre) vs predicted scaffold_platform boxes [(box, conf)]."""
    x, y, w, h = b; fx, fy = x + w / 2, y + h; h = max(h, 1e-3)
    if not plats: return [0., 5., 3., 0., 0., 0., 0.]
    inside, dmin, gap, ov = 0., 5., 3., 0.
    sx1, sx2, sy1, sy2 = x, x + w, y + 0.8 * h, y + h          # lower 20% strip of the person box
    for (px, py, pw, ph), _ in plats:
        if px <= fx <= px + pw and py <= fy <= py + ph: inside = 1.
        dx = max(px - fx, 0, fx - (px + pw)); dy = max(py - fy, 0, fy - (py + ph))
        dmin = min(dmin, (dx * dx + dy * dy) ** .5 / h)
        if px <= fx <= px + pw:                                  # platform under/over the foot x: signed gap to its top
            g = (fy - py) / h
            if abs(g) < abs(gap): gap = float(np.clip(g, -3, 3))
        iw = max(0, min(sx2, px + pw) - max(sx1, px)); ih = max(0, min(sy2, py + ph) - max(sy1, py))
        ov = max(ov, iw * ih / max((sx2 - sx1) * (sy2 - sy1), 1e-6))
    return [inside, min(dmin, 5.), gap, ov, len(plats) / 10, max(c for _, c in plats), 1.]


def structure_geo(b, structs):
    """7 numbers: foot point (box bottom-centre) vs predicted scaffold_structure boxes [(box, conf)].
    inside, distance/h (cap 5), foot height within the structure's vertical extent (0 = structure bottom, 1 = top;
    -1 if no structure spans the foot x), overlap of the person's lower 20% with a structure, n/10, max conf, any."""
    x, y, w, h = b; fx, fy = x + w / 2, y + h; h = max(h, 1e-3)
    if not structs: return [0., 5., -1., 0., 0., 0., 0.]
    inside, dmin, rel, ov = 0., 5., -1., 0.
    sx1, sx2, sy1, sy2 = x, x + w, y + 0.8 * h, y + h
    for (px, py, pw, ph), _ in structs:
        if px <= fx <= px + pw and py <= fy <= py + ph: inside = 1.
        dx = max(px - fx, 0, fx - (px + pw)); dy = max(py - fy, 0, fy - (py + ph))
        dmin = min(dmin, (dx * dx + dy * dy) ** .5 / h)
        if px <= fx <= px + pw: rel = max(rel, float(np.clip((py + ph - fy) / max(ph, 1e-3), -1, 2)))
        iw = max(0, min(sx2, px + pw) - max(sx1, px)); ih = max(0, min(sy2, py + ph) - max(sy1, py))
        ov = max(ov, iw * ih / max((sx2 - sx1) * (sy2 - sy1), 1e-6))
    return [inside, min(dmin, 5.), rel, ov, len(structs) / 10, max(c for _, c in structs), 1.]


def features(cfg, imgs, dets):
    """Feature arrays for every (gid, kind, idx): kind 'gt' = labeled person, 'det' = detected person."""
    path = cfg["_cache_dir"] / "features.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files if k != "keys"}, [tuple(k) for k in z["keys"]]
    enc = Encoder(); pc, plc = cfg["detector"]["person_class"], cfg["detector"]["platform_class"]
    F = {k: [] for k in INPUTS}; keys = []
    for n, g in enumerate(sorted(imgs)):
        t = imgs[g]; dd = dets.get(g, [])
        boxes = [("gt", i, p["box"]) for i, p in enumerate(t["people"])]
        dp = [d for d in dd if d["cls"] == pc]
        boxes += [("det", j, tuple(d["box"])) for j, d in enumerate(dp)]
        if not boxes: continue
        plats = [(tuple(d["box"]), d["conf"]) for d in dd if d["cls"] == plc]
        sc = cfg["detector"].get("structure_class", 5)
        structs = [(tuple(d["box"]), d["conf"]) for d in dd if d["cls"] == sc]
        im = Image.open(t["img"]).convert("RGB")
        c0 = [crop_c0(im, b) for _, _, b in boxes]; c1 = [crop_c1(im, b) for _, _, b in boxes]
        e0, s0 = enc(c0); e1, _ = enc(c1)
        for k, (kind, i, b) in enumerate(boxes):
            n_people = len(t["people"]) if kind == "gt" else len(dp)
            g6 = geo6(b, n_people)
            F["C0"].append(np.concatenate([e0[k], g6])); F["C1"].append(np.concatenate([e1[k], g6]))
            F["C2"].append(np.concatenate([e0[k], g6, support_geo(b, plats)])); F["C3"].append(np.concatenate([s0[k], g6]))
            sg = structure_geo(b, structs)
            F["C0G"].append(np.concatenate([e0[k], g6, sg])); F["C1G"].append(np.concatenate([e1[k], g6, sg]))
            keys.append((g, kind, i))
        if n % 100 == 0: print("features", n, len(imgs), flush=True)
    F = {k: np.array(v, dtype=np.float32) for k, v in F.items()}
    np.savez(path, keys=np.array(keys, dtype=object), **F)
    return F, keys


# ---------------------------------------------------------------- model
def fit(X, y, l2, iters=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y, dtype=torch.long)
    cnt = torch.bincount(yt, minlength=2).float().clamp(min=1); w = cnt.sum() / (2 * cnt)
    torch.manual_seed(0); lin = torch.nn.Linear(X.shape[1], 2)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=iters); lossf = torch.nn.CrossEntropyLoss(weight=w)
    def closure():
        opt.zero_grad(); l = lossf(lin(Xt), yt) + l2 * (lin.weight ** 2).sum(); l.backward(); return l
    opt.step(closure)
    def prob(Z):
        with torch.no_grad(): return torch.softmax(lin(torch.tensor((Z - mu) / sd)), 1)[:, 1].numpy()
    return prob


def ap_score(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.sum() == 0: return float("nan")
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def bal_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(set(y)) < 2: return float("nan")
    return float(((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()) / 2)


def train_draw(imgs, seed, frac):
    tr = sorted(g for g, t in imgs.items() if t["set"] == "train_v2")
    groups = sorted({imgs[g]["group"] for g in tr}); random.Random(seed).shuffle(groups)
    keep = set(groups[:round(frac * len(groups))])
    return [g for g in tr if imgs[g]["group"] in keep]


TARGETS = {"on": lambda p: {"on": 1, "not_on": 0}.get(p["on"]),
           "helmet": lambda p: {"worn": 1, "not_worn": 0}.get(p["helmet"]),
           "harness": lambda p: {"worn": 1, "not_worn": 0}.get(p["harness"])}


def run(cfg, arm, seed, outdir, frac):
    arms = {a["name"]: a for a in cfg["arms"]}
    assert arm in arms, f"arm {arm} not in config arms {list(arms)}"
    imgs = load_people(cfg); dets = detections(cfg, imgs); F, keys = features(cfg, imgs, dets)
    X = F[arm]; row = {k: i for i, k in enumerate(keys)}
    l2 = float(cfg["judge"]["l2"]); thr = cfg["detector"]["match_iou"]; pc = cfg["detector"]["person_class"]
    frac = frac if frac is not None else cfg["repeats"].get("train_fraction_per_draw", 1.0)
    trg = train_draw(imgs, seed, frac)
    models = {}
    for tname, f in TARGETS.items():
        rows = [(row[(g, "gt", i)], f(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f(p) is not None]
        models[tname] = fit(X[[r for r, _ in rows]], np.array([y for _, y in rows]), l2)
        if tname == "on": n_train = (len(rows), sum(y for _, y in rows))
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sel = sorted(g for g, t in imgs.items() if t["set"] in SEL_SETS)
    e2e, cond = [], []; hel, har = ([], []), ([], []); spurious = []; nmiss = 0
    for g in sel:
        t = imgs[g]; dp = [d for d in dets.get(g, []) if d["cls"] == pc]
        gt = [p["box"] for p in t["people"]]; m = match(gt, [tuple(d["box"]) for d in dp], thr)
        det_s = models["on"](X[[row[(g, "det", j)] for j in range(len(dp))]]) if dp else np.zeros(0)
        gt_s = {k: models[k](X[[row[(g, "gt", i)] for i in range(len(gt))]]) if gt else np.zeros(0) for k in models}
        for i, p in enumerate(t["people"]):
            j = m.get(i)
            base = dict(image_id=g, person_id=p["pid"], set=t["set"], on_scaffold=p["on"], group_id=t["group"],
                        review_status=p["review"], support_surface=p["support"])
            e2e.append({**base, "score": round(float(det_s[j]), 6) if j is not None else 0.0, "matched": int(j is not None),
                        "det_conf": dp[j]["conf"] if j is not None else ""})
            cond.append({**base, "score": round(float(gt_s["on"][i]), 6), "matched": 1, "det_conf": ""})
            for k, acc in (("helmet", hel), ("harness", har)):
                yv = TARGETS[k](p)
                if yv is not None: acc[0].append(yv); acc[1].append(int(gt_s[k][i] >= 0.5))
        matched_dets = set(m.values())
        spurious += [float(det_s[j]) for j in range(len(dp)) if j not in matched_dets and all(iou(tuple(dp[j]["box"]), b) < 0.3 for b in gt)]
    cols = ["image_id", "person_id", "set", "on_scaffold", "score", "group_id", "matched", "det_conf", "review_status", "support_surface"]
    tag = f"{arm}_d{seed}" + (f"_f{frac:g}" if frac != cfg["repeats"].get("train_fraction_per_draw", 1.0) else "")
    for name, rows in ((f"scores_{tag}.csv", e2e), (f"scores_cond_{tag}.csv", cond)):
        with open(outdir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    yc = [r["on_scaffold"] == "on" for r in cond if r["on_scaffold"] in ("on", "not_on")]
    sc = [r["score"] for r in cond if r["on_scaffold"] in ("on", "not_on")]
    extra = dict(arm=arm, seed=seed, train_fraction=frac, train_images=len(trg), train_people_on_task=n_train[0],
                 train_positives=n_train[1], conditional_ap=round(ap_score(yc, sc), 4),
                 helmet_bal_acc=round(bal_acc(*hel), 4), helmet_n=len(hel[0]),
                 harness_bal_acc=round(bal_acc(*har), 4), harness_n=len(har[0]),
                 spurious_detections=len(spurious), spurious_scores=sorted(round(s, 4) for s in spurious),
                 config=cfg["_path"])
    json.dump(extra, open(outdir / f"extra_{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in extra.items() if k != "spurious_scores"}))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--config", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", required=True); r.add_argument("--arm", required=True)
    r.add_argument("--seed", type=int, required=True); r.add_argument("--outdir", required=True)
    r.add_argument("--train-frac", type=float, default=None, help="C4 learning-curve hook (default: config value)")
    a = ap.parse_args(); cfg = load_cfg(a.config)
    if a.cmd == "prepare":
        imgs = load_people(cfg); dets = detections(cfg, imgs); F, keys = features(cfg, imgs, dets)
        print("images", len(imgs), "feature rows", len(keys), {k: v.shape for k, v in F.items()})
    else:
        run(cfg, a.arm, a.seed, a.outdir, a.train_frac)


if __name__ == "__main__":
    main()
