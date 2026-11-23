#!/usr/bin/env python3
"""Threshold rule + primary measure (decision_rule.md §3-5). Selection data only: test_v1 rows are refused.

Input CSV (one row per labeled person): image_id, set (val_v2 | cs10k_dev), on_scaffold (on | not_on | unclear |
unannotated), score (judge probability; 0 for an on-scaffold person the detector missed). Optional: group_id
(sequence/site; used for the bootstrap instead of image_id).

  threshold.py scores.csv                      -> cutoffs at FA 1.5% and 5%, recall with counts and bootstrap CI
  threshold.py scores_arm.csv --baseline scores_base.csv
                                               -> paired difference (same people), CI, and the §5 verdict
"""
import argparse, csv, sys
import numpy as np

BUDGETS = (0.015, 0.05)
MIN_GAIN = 0.05


def load(path):
    rows = list(csv.DictReader(open(path)))
    bad = [r for r in rows if r.get("set") == "test_v1"]
    if bad:
        sys.exit("refused: test_v1 rows found. Thresholds and selection use val_v2 + cs10k_dev only.")
    rows = [r for r in rows if r["on_scaffold"] in ("on", "not_on")]   # unclear / unannotated excluded
    key = [f'{r["image_id"]}|{r.get("person_id", i)}' for i, r in enumerate(rows)]
    grp = np.array([r.get("group_id") or r["image_id"] for r in rows])
    y = np.array([r["on_scaffold"] == "on" for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    return key, grp, y, s


def cutoff(y, s, budget):
    """Lowest threshold t (predict on if score >= t) with FA rate on negatives <= budget."""
    neg = np.sort(s[~y])[::-1]
    n = len(neg)
    if n == 0: return np.inf
    k = int(np.floor(budget * n))           # allowed false alarms
    # threshold just above the (k+1)-th highest negative score
    return np.nextafter(neg[k], np.inf) if k < n else -np.inf


def measure(y, s, budget):
    t = cutoff(y, s, budget)
    pred = s >= t
    tp, pos = int((pred & y).sum()), int(y.sum())
    fa, neg = int((pred & ~y).sum()), int((~y).sum())
    return dict(threshold=float(t), recall=tp / pos if pos else float("nan"), tp=tp, pos=pos, fa=fa, neg=neg)


def boot_idx(grp, rng, B):
    ug = np.unique(grp); members = {g: np.where(grp == g)[0] for g in ug}
    for _ in range(B):
        pick = rng.choice(ug, size=len(ug), replace=True)
        yield np.concatenate([members[g] for g in pick])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scores"); ap.add_argument("--baseline")
    ap.add_argument("--resamples", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    key, grp, y, s = load(a.scores)
    if not a.baseline:
        for b in BUDGETS:
            m = measure(y, s, b)
            boots = [measure(y[i], s[i], b)["recall"] for i in boot_idx(grp, rng, a.resamples)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            print(f"FA budget {b:.1%}: threshold {m['threshold']:.4f} | recall {m['recall']:.3f} "
                  f"({m['tp']} of {m['pos']}) 95% CI [{lo:.3f}, {hi:.3f}] | false alarms {m['fa']} of {m['neg']}")
        return
    kb, gb, yb, sb = load(a.baseline)
    assert kb == key and (yb == y).all(), "paired comparison needs the same people in the same order"
    for b in BUDGETS:
        d0 = measure(y, s, b)["recall"] - measure(yb, sb, b)["recall"]
        diffs = [measure(y[i], s[i], b)["recall"] - measure(yb[i], sb[i], b)["recall"] for i in boot_idx(grp, rng, a.resamples)]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        verdict = "REPLACE baseline" if (d0 >= MIN_GAIN and lo > 0) else "KEEP baseline"
        print(f"FA {b:.1%}: recall diff {d0:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}"
              + ("  (primary)" if b == BUDGETS[0] else ""))


if __name__ == "__main__":
    main()
