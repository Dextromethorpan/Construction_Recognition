#!/usr/bin/env python3
"""Study-level paired comparison (decision_rule.md §5): mean over draws of (arm - baseline) recall at FA 1.5% / 5%.
The bootstrap resamples images (group_id) once per replicate and applies the same resample to every draw, re-choosing the
threshold inside each resample (threshold.py helpers). Selection data only (threshold.load refuses test rows).

  compare.py DIR --arm C2 --baseline C0 [--seeds 0 1 2 3 4] [--resamples 2000]
"""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import threshold as T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--arm", required=True)
    ap.add_argument("--baseline", default="C0"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--resamples", type=int, default=2000); ap.add_argument("--prefix", default="scores")
    a = ap.parse_args(); d = Path(a.dir)
    A = [T.load(d / f"{a.prefix}_{a.arm}_d{s}.csv") for s in a.seeds]
    B = [T.load(d / f"{a.prefix}_{a.baseline}_d{s}.csv") for s in a.seeds]
    key, grp, y = A[0][0], A[0][1], A[0][2]
    for x in A + B: assert x[0] == key and (x[2] == y).all(), "all files must list the same people in the same order"
    rng = np.random.default_rng(0)
    idx = list(T.boot_idx(grp, rng, a.resamples))
    for b in T.BUDGETS:
        ra = [T.measure(y, x[3], b) for x in A]; rb = [T.measure(y, x[3], b) for x in B]
        d0 = np.mean([p["recall"] - q["recall"] for p, q in zip(ra, rb)])
        diffs = [np.mean([T.measure(y[i], x[3][i], b)["recall"] - T.measure(y[i], z[3][i], b)["recall"] for x, z in zip(A, B)])
                 for i in idx]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        verdict = "REPLACE baseline" if (d0 >= T.MIN_GAIN and lo > 0) else "KEEP baseline"
        print(f"FA {b:.1%}: {a.arm} {np.mean([p['recall'] for p in ra]):.3f} vs {a.baseline} {np.mean([q['recall'] for q in rb]):.3f} | "
              f"mean diff {d0:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}] -> {verdict}" + ("  (primary)" if b == T.BUDGETS[0] else ""))


if __name__ == "__main__":
    main()
