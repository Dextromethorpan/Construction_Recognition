"""H1_harness_crop driver: torso-crop features for labeled people, 5 draws per arm, harness balanced accuracy, paired bootstrap."""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
H = Path.home(); E = Path(__file__).parent; C = H / "GenIA-code"; sys.path.insert(0, str(C / "08_controlled"))
import judge
CFG = E / "config.yaml"; D0 = C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"
cfg = judge.load_cfg(D0); imgs = judge.load_people(cfg)
F0, keys = judge.features(cfg, imgs, judge.detections(cfg, imgs)); row = {k: i for i, k in enumerate(keys)}

def torso(im, b):
    W, Hh = im.size; x, y, w, h = b; cx = (x + w / 2) / 100 * W
    x1, x2 = max(0, cx - 0.6 * w / 100 * W), min(W, cx + 0.6 * w / 100 * W)
    y1, y2 = max(0, (y + 0.15 * h) / 100 * Hh), min(Hh, (y + 0.75 * h) / 100 * Hh)
    c = im.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2))); s = 224 / max(c.size)
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s))))
    cv = Image.new("RGB", (224, 224), (114, 114, 114)); cv.paste(c, ((224 - c.size[0]) // 2, (224 - c.size[1]) // 2)); return cv

cache = E / "torso_features.npz"
if cache.exists():
    z = np.load(cache, allow_pickle=True); TK = [tuple(k) for k in z["keys"]]; TF = z["F"]
else:
    enc = judge.Encoder(); TK, TF = [], []
    for g in sorted(imgs):
        t = imgs[g]
        if not t["people"]: continue
        im = Image.open(t["img"]).convert("RGB"); e, _ = enc([torso(im, p["box"]) for p in t["people"]])
        for i, p in enumerate(t["people"]):
            TK.append((g, "gt", i)); TF.append(np.concatenate([e[i], judge.geo6(p["box"], len(t["people"]))]))
    TF = np.array(TF, dtype=np.float32); np.savez(cache, keys=np.array(TK, dtype=object), F=TF)
trow = {k: i for i, k in enumerate(TK)}
def X(arm, ks):
    a = F0["C0"][[row[k] for k in ks]]; b = TF[[trow[k] for k in ks]]
    return {"H0": a, "H1": b, "H2": np.concatenate([a, b[:, :-6]], 1)}[arm]

sel = [(g, i, p) for g in sorted(imgs) if imgs[g]["set"] in judge.SEL_SETS for i, p in enumerate(imgs[g]["people"])]
results = {}
for tgt in ("harness", "helmet"):
    f = judge.TARGETS[tgt]; sl = [(g, i, f(p)) for g, i, p in sel if f(p) is not None]
    ks = [(g, "gt", i) for g, i, _ in sl]; y = np.array([v for *_, v in sl]); grp = np.array([imgs[g]["group"] for g, *_ in sl])
    for arm in ("H0", "H1", "H2"):
        P = []
        for d in range(5):
            trg = judge.train_draw(imgs, d, 0.8)
            tr = [((g, "gt", i), f(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f(p) is not None]
            m = judge.fit(X(arm, [k for k, _ in tr]), np.array([v for _, v in tr]), 1e-3); P.append(m(X(arm, ks)))
        results[(tgt, arm)] = P
    rng = np.random.default_rng(0); ug = np.unique(grp); mem = {u: np.where(grp == u)[0] for u in ug}
    idx = [np.concatenate([mem[u] for u in rng.choice(ug, len(ug))]) for _ in range(2000)]
    ba = lambda yy, pp: judge.bal_acc(yy, (pp >= 0.5).astype(int))
    print(f"\n== {tgt}: n {len(y)} (worn {int(y.sum())}, not_worn {int((1 - y).sum())})")
    for arm in ("H0", "H1", "H2"):
        P = results[(tgt, arm)]
        print(f"{arm}: bal acc {np.mean([ba(y, p) for p in P]):.3f} (draws {', '.join(f'{ba(y, p):.3f}' for p in P)}) | AP {np.mean([judge.ap_score(y, p) for p in P]):.3f}")
    for arm in ("H1", "H2"):
        A, B = results[(tgt, arm)], results[(tgt, "H0")]
        d0 = np.mean([ba(y, a) - ba(y, b) for a, b in zip(A, B)])
        diffs = [np.mean([ba(y[i], a[i]) - ba(y[i], b[i]) for a, b in zip(A, B)]) for i in idx]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        print(f"{arm} vs H0: diff {d0:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] -> {'REPLACE' if d0 >= 0.05 and lo > 0 else 'KEEP H0'}")
    json.dump({f"{tgt}_{arm}_d{d}": [float(v) for v in results[(tgt, arm)][d]] for arm in ("H0", "H1", "H2") for d in range(5)},
              open(E / f"probs_{tgt}.json", "w"))
