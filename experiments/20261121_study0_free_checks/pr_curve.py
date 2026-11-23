"""Study 0: PR curves of the C0 judge (draw 0) on the selection set (val_v2 + cs10k_dev, real only). Not a decision run."""
import csv, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "../20261121_context_C0_C3/"
def load(f, only=None):
    R = [r for r in csv.DictReader(open(S + f)) if r["on_scaffold"] in ("on", "not_on") and (only is None or r["set"] == only)]
    return np.array([r["on_scaffold"] == "on" for r in R]), np.array([float(r["score"]) for r in R])
def pr(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); fp = np.cumsum(~y)
    return tp / y.sum(), tp / (tp + fp), float((tp / np.arange(1, len(y) + 1) * y).sum() / y.sum())
fig, ax = plt.subplots(figsize=(6, 4.5))
for f, lab in (("scores_cond_C0_d0.csv", "labeled boxes (conditional)"), ("scores_C0_d0.csv", "end-to-end (detector misses = 0)")):
    for only in (None, "val_v2", "cs10k_dev"):
        y, s = load(f, only); r, p, ap = pr(y, s)
        print(f"{lab:35s} {only or 'val_v2+cs10k_dev':17s} pos {int(y.sum()):3d} neg {int((~y).sum()):4d} AP {ap:.3f} max recall with score>0 {((s>0)&y).sum()/y.sum():.3f}")
        if only is None: ax.plot(r, p, label=f"{lab}, AP {ap:.2f}")
ax.set_xlabel("on-scaffold recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.set_title("C0 judge, draw 0, val_v2 + cs10k_dev (real)"); ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig("pr_C0_d0.png", dpi=150)
