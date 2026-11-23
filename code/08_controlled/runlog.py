#!/usr/bin/env python3
"""Append-only run log for controlled experiments: GenIA-code/experiments/RUNLOG.csv.

  runlog.py add --study S --arm A --draw 0 --config experiments/<d>_<S>/config.yaml --label-version v1 \
                --status ok --primary 0.62 --ci 0.55 0.69 --fa-budget 0.015 --threshold 0.41 --n-pos 50 --n-neg 620 [--notes ...]
  runlog.py add ... --status failed --notes "OOM at epoch 3"         (failed / abandoned runs are logged too)
  runlog.py decide --study S --winner A --notes "beats baseline by 0.07, CI [0.02, 0.12]"
  runlog.py check-test --study S       exit 0 only if a decision exists and no test row exists yet
  runlog.py add --study S --arm A --status test ...                  (refused unless check-test passes)
Rows are never edited or deleted.
"""
import argparse, csv, datetime, hashlib, os, sys
from pathlib import Path

C = Path(os.environ.get("GENIA_CODE", Path.home() / "GenIA-code"))
LOG = C / "experiments/RUNLOG.csv"
COLS = ["run_id", "timestamp", "study", "arm", "draw", "seed", "config_path", "config_hash", "label_version",
        "split_version", "series", "status", "primary_metric", "primary_value", "ci_low", "ci_high", "fa_budget",
        "threshold", "n_pos", "n_neg", "kept", "notes"]
STATUSES = {"ok", "failed", "abandoned", "decision", "test", "rule_change"}


def rows():
    return list(csv.DictReader(open(LOG))) if LOG.exists() else []


def config_hash(p):
    if not p: return ""
    path = Path(p) if Path(p).is_absolute() else C / p
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def append(r):
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new: w.writeheader()
        r = {k: r.get(k, "") for k in COLS}
        r["run_id"] = f"r{len(rows()) + 1:05d}"
        r["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        w.writerow(r)
    return r["run_id"]


def can_test(study):
    rs = [r for r in rows() if r["study"] == study]
    if not any(r["status"] == "decision" for r in rs):
        return False, f"no decision recorded for study '{study}' (runlog.py decide first)"
    if any(r["status"] == "test" for r in rs):
        return False, f"study '{study}' already used its single test run"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    for k in ("study", "arm", "status"): a.add_argument(f"--{k}", required=True)
    for k in ("draw", "seed", "config", "label-version", "split-version", "series", "primary-metric",
              "primary", "fa-budget", "threshold", "n-pos", "n-neg", "kept", "notes"):
        a.add_argument(f"--{k}", default="")
    a.add_argument("--ci", nargs=2, default=["", ""])
    d = sub.add_parser("decide"); d.add_argument("--study", required=True); d.add_argument("--winner", required=True)
    d.add_argument("--notes", default="")
    t = sub.add_parser("check-test"); t.add_argument("--study", required=True)
    x = ap.parse_args()
    if x.cmd == "check-test":
        ok, msg = can_test(x.study); print(msg); sys.exit(0 if ok else 1)
    if x.cmd == "decide":
        print(append(dict(study=x.study, arm=x.winner, status="decision", kept="yes", notes=x.notes))); return
    assert x.status in STATUSES, f"status must be one of {sorted(STATUSES)}"
    if x.status == "test":
        ok, msg = can_test(x.study)
        if not ok: sys.exit("refused: " + msg)
    if x.status == "decision":
        sys.exit("use 'runlog.py decide'")
    rid = append(dict(study=x.study, arm=x.arm, draw=x.draw, seed=x.seed or x.draw, config_path=x.config,
                      config_hash=config_hash(x.config), label_version=x.label_version,
                      split_version=x.split_version or "v2", series=x.series, status=x.status,
                      primary_metric=x.primary_metric or "recall_at_fa_0.015_end_to_end", primary_value=x.primary,
                      ci_low=x.ci[0], ci_high=x.ci[1], fa_budget=x.fa_budget, threshold=x.threshold,
                      n_pos=x.n_pos, n_neg=x.n_neg, kept=x.kept, notes=x.notes))
    print(rid)


if __name__ == "__main__":
    main()
