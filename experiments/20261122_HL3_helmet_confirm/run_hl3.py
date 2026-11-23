import json, sys, random, numpy as np
from pathlib import Path
from PIL import Image
H = Path.home(); C = H / "GenIA-code"; E = Path(__file__).parent; D = H / "GenIA-data"
sys.path.insert(0, str(C / "08_controlled")); import judge
sys.path.insert(0, str(C / "experiments/20261122_V3_confirm"))
pool = json.load(open(E / "pool.json")); lab = dict(l.split() for l in open(E / "labels.txt"))
N = [(g, b, 0) for i, (g, b) in enumerate(pool["viol"]) if lab.get(str(i)) == "N"]
Wv = [(g, b, 1) for i, (g, b) in enumerate(pool["viol"]) if lab.get(str(i)) == "W"]
Ws = pool["white"][:2 * len(N)]; people = N + Wv + [(g, b, 1) for g, b in Ws]
print("not_worn", len(N), "worn", len(Wv) + len(Ws), flush=True)
def head_crop(im, b):
    W, Hh = im.size; x, y, w, h = b; cx = (x + w / 2) / 100 * W
    x1, x2 = max(0, cx - 0.5 * w / 100 * W), min(W, cx + 0.5 * w / 100 * W); y1, y2 = max(0, (y - 0.15 * h) / 100 * Hh), min(Hh, (y + 0.35 * h) / 100 * Hh)
    c = im.crop((x1, y1, max(x2, x1 + 2), max(y2, y1 + 2))); s = 224 / max(c.size)
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s)))); cv = Image.new("RGB", (224, 224), (114, 114, 114))
    cv.paste(c, ((224 - c.size[0]) // 2, (224 - c.size[1]) // 2)); return cv
enc = judge.Encoder(); X0, XH, y, grp = [], [], [], []
for g, (bx, by_, bw, bh), lbl in people:
    im = Image.open(D / "extracted/cs10k" / (g.split("/")[1] + ".jpg")).convert("RGB"); W, Hh = im.size
    b = (bx / W * 100, by_ / Hh * 100, bw / W * 100, bh / Hh * 100)
    e0, _ = enc([judge.crop_c0(im, b)]); eh, _ = enc([head_crop(im, b)])
    X0.append(np.concatenate([e0[0], judge.geo6(b, 1)])); XH.append(eh[0]); y.append(lbl); grp.append(g)
X0, XH, y, grp = np.array(X0, np.float32), np.array(XH, np.float32), np.array(y), np.array(grp)
cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); timgs = judge.load_people(cfg)
F, keys = judge.features(cfg, timgs, judge.detections(cfg, timgs)); row = {k: i for i, k in enumerate(keys)}
z = np.load(C / "experiments/20261122_HL1_helmet_crop/head_features.npz", allow_pickle=True); HK = {tuple(k): i for i, k in enumerate(z["keys"])}; HF = z["F"]
p0, p2 = [], []
for d in range(5):
    trg = judge.train_draw(timgs, d, 0.8); hr = [((g, "gt", i), judge.TARGETS["helmet"](p)) for g in trg for i, p in enumerate(timgs[g]["people"]) if judge.TARGETS["helmet"](p) is not None]
    A0 = F["C0"][[row[k] for k, _ in hr]]; AH = HF[[HK[k] for k, _ in hr]]; yh = np.array([v for _, v in hr])
    p0.append(judge.fit(A0, yh, 1e-3)(X0)); p2.append(judge.fit(np.concatenate([A0, AH[:, :-6]], 1), yh, 1e-3)(np.concatenate([X0, XH], 1)))
ba = lambda yy, pp: judge.bal_acc(yy, (pp >= 0.5).astype(int))
A = np.mean([ba(y, p) for p in p2]); B = np.mean([ba(y, p) for p in p0])
ug = np.unique(grp); mem = {u: np.where(grp == u)[0] for u in ug}; rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    ix = np.concatenate([mem[u] for u in rng.choice(ug, len(ug))]); diffs.append(np.mean([ba(y[ix], a[ix]) - ba(y[ix], b[ix]) for a, b in zip(p2, p0)]))
lo, hi = np.nanpercentile(diffs, [2.5, 97.5]); nw = lambda P: np.mean([np.mean(p[y == 0] < 0.5) for p in P]); wc = lambda P: np.mean([np.mean(p[y == 1] >= 0.5) for p in P])
print(f"C0 {B:.3f} vs C0+head {A:.3f} | diff {A - B:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] | not_worn caught {nw(p0):.3f} -> {nw(p2):.3f} | worn correct {wc(p0):.3f} -> {wc(p2):.3f}")
