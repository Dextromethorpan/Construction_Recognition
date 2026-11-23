#!/usr/bin/env python3
"""Fit the FINAL pipeline's draw-0 heads (exactly as experiments/20261123_final_test/test_final.py, d=0) from the cached
train features and save them as plain arrays to 09_demo/heads_d0.npz, so demo.py needs no training caches.
judge.fit returns a closure; fit_arrays below is the same recipe (same seed, LBFGS, weights, l2) returning arrays.
Verifies the saved arrays reproduce judge.fit's closure probabilities.
  .venv-train/bin/python 09_demo/fit_heads.py"""
import sys
from pathlib import Path
import numpy as np, torch
C = Path.home() / "GenIA-code"; OUT = Path(__file__).parent / "heads_d0.npz"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "08_controlled"))
import judge, threshold as T


def fit_arrays(X, y, l2, iters=300):          # = judge.fit, but returns (mu, sd, W, b)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y, dtype=torch.long)
    cnt = torch.bincount(yt, minlength=2).float().clamp(min=1); w = cnt.sum() / (2 * cnt)
    torch.manual_seed(0); lin = torch.nn.Linear(X.shape[1], 2)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=iters); lossf = torch.nn.CrossEntropyLoss(weight=w)
    def closure():
        opt.zero_grad(); l = lossf(lin(Xt), yt) + l2 * (lin.weight ** 2).sum(); l.backward(); return l
    opt.step(closure)
    return mu, sd, lin.weight.detach().numpy().copy(), lin.bias.detach().numpy().copy()


def prob_np(Z, mu, sd, W, b):
    z = ((Z - mu) / sd) @ W.T + b; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return (e[:, 1] / e.sum(1))


cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
F, keys = judge.features(cfg, imgs, judge.detections(cfg, imgs)); row = {k: i for i, k in enumerate(keys)}
trg = judge.train_draw(imgs, 0, 0.8); save = {}
for tname, arm in (("on", "C1"), ("helmet", "C0"), ("harness", "C0")):
    f = judge.TARGETS[tname]; rr = [(row[(g, "gt", i)], f(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f(p) is not None]
    X, y = F[arm][[r for r, _ in rr]], np.array([y for _, y in rr])
    clo = judge.fit(X, y, 1e-3); mu, sd, W, b = fit_arrays(X, y, 1e-3)
    probe = F[arm][np.random.default_rng(0).choice(len(F[arm]), 200, replace=False)]
    diff = np.abs(clo(probe) - prob_np(probe, mu, sd, W, b)).max()
    print(f"{tname}: arm {arm} n_train {len(y)} pos {int(y.sum())} max|closure - saved| on 200 people = {diff:.2e}")
    assert diff < 1e-5
    save.update({f"{tname}_mu": mu, f"{tname}_sd": sd, f"{tname}_W": W, f"{tname}_b": b})
_, _, yv, sv = T.load(C / "experiments/20261121_D1_detector/scores_D0_d0.csv")
save["on_cut"] = np.float64(T.cutoff(yv, sv, 0.015)); save["on_cut_5pct"] = np.float64(T.cutoff(yv, sv, 0.05))
print("on-scaffold cutoff FA1.5%:", float(save["on_cut"]), " FA5%:", float(save["on_cut_5pct"]))
np.savez(OUT, **save); print("saved", OUT)
