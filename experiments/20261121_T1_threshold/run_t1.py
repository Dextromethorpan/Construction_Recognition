import sys, numpy as np
from pathlib import Path
C = Path.home() / "GenIA-code"; sys.path.insert(0, str(C / "08_controlled")); import threshold as T
res = {r: {"fa": [], "rec": []} for r in ("T0", "T1", "T2")}
for d in range(5):
    key, grp, y, s = T.load(C / f"experiments/20261121_D1_detector/scores_D0_d{d}.csv")
    ug = np.unique(grp); rng = np.random.default_rng(d)
    for it in range(100):
        A = set(rng.choice(ug, len(ug) // 2, replace=False)); a = np.array([g in A for g in grp]); b = ~a
        cuts = {"T0": T.cutoff(y[a], s[a], 0.015), "T2": T.cutoff(y[a], s[a], 0.0075)}
        ia = np.where(a)[0]; ga = grp[a]; bs = []
        for idx in T.boot_idx(ga, rng, 200): bs.append(T.cutoff(y[ia][idx], s[ia][idx], 0.015))
        cuts["T1"] = np.percentile(bs, 90)
        for r, t in cuts.items():
            pred = s[b] >= t; res[r]["fa"].append(pred[~y[b]].mean()); res[r]["rec"].append(pred[y[b]].mean())
for r, v in res.items():
    fa, rc = np.array(v["fa"]), np.array(v["rec"])
    print(f"{r}: realised FA median {np.median(fa):.2%} p90 {np.percentile(fa, 90):.2%} mean {fa.mean():.2%} | recall mean {rc.mean():.3f}")
