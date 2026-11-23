"""Ablation design B: per-person second-stage classifiers, same label-policy variants.

Features per person = yolo11n-cls embedding of the person crop with context (box widened 2x, heightened 1.5x)
+ 6 box-geometry numbers. Classifier = class-balanced L2 logistic regression (torch), identical for all variants.
Dev pool = all relation-labeled pass-1 images except CS10k (train, val, test_indist, test_heldout).
Scoring = 5-fold CV on REAL dev people (synthetic people only ever used for training); folds grouped by image,
and by whole source for video-frame sources (rf_scaffold_mobile, rf_harness_mks) to avoid near-duplicate leakage.
Holdout = CS10k (test_external): train on the whole dev pool, score once.
GT person boxes are used as input, so this isolates the label policy from person detection."""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import json, csv, random, hashlib
from pathlib import Path
import numpy as np, torch
from PIL import Image
from ultralytics import YOLO

D = Path(GENIA_DATA); REL = D / "labels/relations/v1"
OUT = D / "runs/ablation_B"; OUT.mkdir(parents=True, exist_ok=True)
VIDEO_SRC = {"rf_scaffold_mobile", "rf_harness_mks"}

def people_of(result):
    P, other = {}, []
    for r in result:
        v = r.get("value", {})
        if r.get("type") == "rectanglelabels":
            if v["rectanglelabels"][0] == "person": P.setdefault(r["id"], {})["box"] = (v["x"], v["y"], v["width"], v["height"])
            elif v["rectanglelabels"][0] == "harness": other.append((v["x"], v["y"], v["width"], v["height"]))
        elif r.get("type") == "choices" and r.get("from_name") in ("support", "helmet_state", "harness_state"):
            P.setdefault(r["id"], {})[r["from_name"]] = v["choices"][0]
    P = [p for p in P.values() if "box" in p]
    for p in P:   # harness-from-boxes target: a harness box mostly inside the person box
        b = p["box"]
        def ioa(s):
            iw = max(0, min(s[0]+s[2], b[0]+b[2]) - max(s[0], b[0])); ih = max(0, min(s[1]+s[3], b[1]+b[3]) - max(s[1], b[1]))
            return iw*ih/(s[2]*s[3]) if s[2]*s[3] else 0
        p["harness_box"] = any(ioa(s) >= 0.5 for s in other)
    return P

def load(fn):
    out = {}
    for t in json.load(open(fn)):
        d = t["data"]
        if d.get("pass") != 1 or not t["annotations"]: continue
        res = t["annotations"][-1]["result"]
        if any("bad_image" in c for r in res if r.get("from_name") == "image_flag" for c in r["value"]["choices"]): continue
        out[d["global_id"]] = {"split": d["split"], "img": str(D / d["image"].split("?d=")[1]), "people": people_of(res)}
    return out

def iou(a, b):
    iw = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); ih = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = iw*ih; u = a[2]*a[3] + b[2]*b[3] - i
    return i/u if u else 0

_emb_cache = {}
def featurize(tasks, model, tag):
    cache = OUT / f"feat_{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); return z["X"], list(z["keys"])
    X, keys = [], []
    for gid, t in sorted(tasks.items()):
        im = Image.open(t["img"]).convert("RGB"); W, H = im.size
        crops, geo = [], []
        for i, p in enumerate(t["people"]):
            x, y, w, h = p["box"]
            cx, cy = x + w/2, y + h/2
            x1, x2 = max(0, cx - w), min(100, cx + w); y1, y2 = max(0, cy - 0.75*h), min(100, cy + 0.75*h)
            crops.append(im.crop((x1/100*W, y1/100*H, max(x2/100*W, x1/100*W+2), max(y2/100*H, y1/100*H+2))).resize((224, 224)))
            geo.append([cx/100, cy/100, w/100, h/100, (y+h)/100, len(t["people"])/10])
            keys.append((gid, i))
        if crops:
            E = [e.cpu().numpy() for e in model.embed(crops, imgsz=224, verbose=False)]
            X += [np.concatenate([e, g]) for e, g in zip(E, geo)]
    X = np.array(X, dtype=np.float32); np.savez(cache, X=X, keys=np.array(keys, dtype=object))
    return X, keys

def fit_predict(Xtr, ytr, Xte, n_cls, l2=1e-3, epochs=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = torch.tensor((Xtr-mu)/sd), torch.tensor((Xte-mu)/sd); ytr = torch.tensor(ytr)
    cnt = torch.bincount(ytr, minlength=n_cls).float().clamp(min=1); wts = cnt.sum()/(n_cls*cnt)
    torch.manual_seed(0); lin = torch.nn.Linear(Xtr.shape[1], n_cls)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=epochs)
    lossf = torch.nn.CrossEntropyLoss(weight=wts)
    def closure():
        opt.zero_grad(); l = lossf(lin(Xtr), ytr) + l2*(lin.weight**2).sum(); l.backward(); return l
    opt.step(closure)
    with torch.no_grad(): return lin(Xte).argmax(1).numpy()

def metrics(y, p, ci=True):
    y, p = np.array(y), np.array(p)
    if ci and len(y):   # person-level bootstrap 95% CI for F1 and balanced accuracy
        rng = np.random.default_rng(0); f1s, bas = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y)); m = metrics(y[i], p[i], ci=False); f1s.append(m["F1"]); bas.append(m["bal_acc"])
        out = metrics(y, p, ci=False)
        out["F1_ci95"] = [round(float(np.percentile(f1s, 2.5)), 3), round(float(np.percentile(f1s, 97.5)), 3)]
        out["bal_acc_ci95"] = [round(float(np.percentile(bas, 2.5)), 3), round(float(np.percentile(bas, 97.5)), 3)]
        return out
    tp, fp, fn = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    tpr = rc; tnr = ((p == 0) & (y == 0)).sum()/max(1, (y == 0).sum())
    return {"n": len(y), "pos": int(y.sum()), "P": round(pr, 4), "R": round(rc, 4),
            "F1": round(2*pr*rc/(pr+rc), 4) if pr+rc else 0.0, "bal_acc": round(float(tpr+tnr)/2, 4),
            "fp_rate": round(float(((p == 1) & (y == 0)).sum()/max(1, (y == 0).sum())), 4)}

# --- targets per variant -------------------------------------------------------------------------
ON = {"stands_on": 1, "climbing": 1, "none": 0}
def on_target(s, var):            # returns train label or None (=drop); 2 = unclear class
    if s in ON: return ON[s]
    if var == "V1_unclear_class": return 2
    if var == "V5_unclear_neg": return 0
    return None
def harness_target(p, var):
    if var == "V2_harness_field":
        return {"harness:worn": 1, "harness:not_worn": 0}.get(p.get("harness_state"))
    return int(p["harness_box"])
HELMET = {"helmet:worn": 1, "helmet:not_worn": 0}

def main():
    codex, claude = load(REL / "frozen_v1/codex_reviewed.json"), load(REL / "export_20261118_220544.json")
    synth = {r["global_id"] for r in csv.DictReader(open(REL / "synthetic_images.csv"))}
    emb = YOLO("yolo11n-cls.pt")
    Xc, kc = featurize(codex, emb, "codex"); Xl, kl = featurize(claude, emb, "claude")
    rowc = {k: i for i, k in enumerate(map(tuple, kc))}; rowl = {k: i for i, k in enumerate(map(tuple, kl))}
    dev = [g for g, t in codex.items() if t["split"] != "test_external"]
    hold = [g for g, t in codex.items() if t["split"] == "test_external"]
    # folds: group = image, or whole source for video sources; real images only are scored
    groups = sorted({g.split("/")[0] if g.split("/")[0] in VIDEO_SRC else g for g in dev})
    random.Random(0).shuffle(groups); fold_of_group = {gr: i % 5 for i, gr in enumerate(groups)}
    fold = {g: fold_of_group[g.split("/")[0] if g.split("/")[0] in VIDEO_SRC else g] for g in dev}

    # Claude people -> matched to Codex people (IoU>=0.5) only for scoring; training uses Claude's own boxes+labels
    def train_rows(var, gids):
        src, X, rows = (claude, Xl, rowl) if var == "V3_claude" else (codex, Xc, rowc)
        out = []
        for g in gids:
            if var == "V4_no_synth" and g in synth: continue
            if g not in src: continue
            for i, p in enumerate(src[g]["people"]):
                out.append((X[rows[(g, i)]], p))
        return out
    def eval_rows(gids):
        return [(Xc[rowc[(g, i)]], p) for g in gids if g not in synth for i, p in enumerate(codex[g]["people"])]

    variants = ["V0_base", "V1_unclear_class", "V2_harness_field", "V3_claude", "V4_no_synth", "V5_unclear_neg"]
    results = {}
    for var in variants:
        R = {}
        for task in ("on_scaffold", "helmet", "harness"):
            def lab_tr(p):
                if task == "on_scaffold": return on_target(p.get("support"), var)
                if task == "helmet": return HELMET.get(p.get("helmet_state"))
                return harness_target(p, var)
            def lab_ev(p):        # evaluation truth is always Codex labels; harness truth = per-person field
                if task == "on_scaffold": return ON.get(p.get("support"))
                if task == "helmet": return HELMET.get(p.get("helmet_state"))
                return {"harness:worn": 1, "harness:not_worn": 0}.get(p.get("harness_state"))
            def run(tr_g, te_g):
                tr = [(x, lab_tr(p)) for x, p in train_rows(var, tr_g)]; tr = [(x, y) for x, y in tr if y is not None]
                te = [(x, lab_ev(p)) for x, p in eval_rows(te_g)]; te = [(x, y) for x, y in te if y is not None]
                if not tr or not te: return [], []
                n_cls = 3 if (task == "on_scaffold" and var == "V1_unclear_class") else 2
                pred = fit_predict(np.stack([x for x, _ in tr]), np.array([y for _, y in tr]), np.stack([x for x, _ in te]), n_cls)
                pred = np.where(pred == 2, 0, pred)   # predicted 'unclear' counts as not on scaffold
                return [y for _, y in te], list(pred)
            ys, ps = [], []
            for f in range(5):
                y, p = run([g for g in dev if fold[g] != f], [g for g in dev if fold[g] == f]); ys += y; ps += p
            R[task] = {"cv_dev_real": metrics(ys, ps), "cs10k_holdout": metrics(*run(dev, hold))}
        results[var] = R
        print(var, {t: (R[t]["cv_dev_real"]["F1"], R[t]["cv_dev_real"]["bal_acc"], R[t]["cs10k_holdout"]["fp_rate"]) for t in R}, flush=True)
    json.dump(results, open(OUT / "scores.json", "w"), indent=1)

if __name__ == "__main__":
    main()
