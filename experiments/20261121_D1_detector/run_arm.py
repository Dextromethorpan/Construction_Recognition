"""D1_detector driver: run_arm.py ARM WEIGHTS IMGSZ [--dets PATH]
Derived judge config per arm (main config + detector override, judge input C1), judge prepare + 5 draws,
threshold.py, runlog.py add. Scores land in this folder as scores_<ARM>_d<k>.csv for compare.py. Selection data only."""
import json, re, shutil, subprocess, sys, argparse
from pathlib import Path
E = Path(__file__).parent; C = E.parent.parent; H = Path.home(); PY = str(C / ".venv-train/bin/python"); K = C / "08_controlled"
CFG = f"experiments/{E.name}/config.yaml"; STUDY = "D1_detector"
sys.path.insert(0, str(C / ".venv-train/lib")); from ultralytics.utils import YAML
ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("weights"); ap.add_argument("imgsz", type=int)
ap.add_argument("--dets", help="precomputed detections.json (e.g. tiled inference)"); ap.add_argument("--study", default=STUDY)
ap.add_argument("--config", default=CFG)
a = ap.parse_args(); STUDY, CFG = a.study, a.config
cfg = YAML.load(str(E / "config.yaml")); cache = H / f"GenIA-data/runs/controlled/{STUDY}/{a.arm}"; cache.mkdir(parents=True, exist_ok=True)
if a.dets and not (cache / "detections.json").exists(): shutil.copy(a.dets, cache / "detections.json")
cfg["detector"].update(weights=a.weights, imgsz=a.imgsz, cache=str(cache / "detections.json"))
cfg["judge"]["feature_cache"] = str(cache / "features.npz"); cfg["arms"] = [{"name": "C1", "change": f"detector arm {a.arm}"}]
O = C / Path(CFG).parent; dc = O / "arm_configs"; dc.mkdir(exist_ok=True); acfg = dc / f"{a.arm}.yaml"; json.dump(cfg, open(acfg, "w"), indent=1)
subprocess.run([PY, K / "judge.py", "prepare", "--config", acfg], cwd=C, check=True)
out_dir = O / f"arm_{a.arm}"
for d in range(5):
    base = ["--study", STUDY, "--arm", a.arm, "--draw", str(d), "--config", CFG, "--label-version", "v2",
            "--split-version", "v2", "--series", "real_only", "--fa-budget", "0.015"]
    try:
        subprocess.run([PY, K / "judge.py", "run", "--config", acfg, "--arm", "C1", "--seed", str(d), "--outdir", out_dir],
                       cwd=C, check=True, capture_output=True, text=True)
        for pre in ("scores", "scores_cond", "extra"):
            ext = "json" if pre == "extra" else "csv"
            shutil.copy(out_dir / f"{pre}_C1_d{d}.{ext}", O / f"{pre}_{a.arm}_d{d}.{ext}")
        out = subprocess.run([PY, K / "threshold.py", O / f"scores_{a.arm}_d{d}.csv"], cwd=C, check=True, capture_output=True, text=True).stdout
        (O / f"threshold_{a.arm}_d{d}.txt").write_text(out)
        m = re.search(r"FA budget 1.5%: threshold (\S+) \| recall (\S+) \((\d+) of (\d+)\) 95% CI \[(\S+), (\S+)\] \| false alarms (\d+) of (\d+)", out)
        m5 = re.search(r"FA budget 5.0%: threshold \S+ \| recall (\S+) \((\d+) of \d+\) 95% CI \[(\S+), (\S+)\]", out)
        x = json.load(open(O / f"extra_{a.arm}_d{d}.json"))
        import csv
        rows = [r for r in csv.DictReader(open(O / f"scores_{a.arm}_d{d}.csv")) if r["on_scaffold"] == "on"]
        boxed = sum(int(r["matched"]) for r in rows); thr = float(m.group(1))
        spur = sum(s >= thr for s in x["spurious_scores"])
        notes = (f"weights {Path(a.weights).parent.parent.name} imgsz {a.imgsz}; on-scaffold boxed {boxed} of {len(rows)}; "
                 f"FA {m.group(7)} of {m.group(8)}; recall@5% {m5.group(1)} ({m5.group(2)} of {m.group(4)}); "
                 f"spurious dets {x['spurious_detections']} ({spur} above cutoff); cond AP {x['conditional_ap']}; "
                 f"helmet bal {x['helmet_bal_acc']}; harness bal {x['harness_bal_acc']}")
        args = base + ["--status", "ok", "--primary", m.group(2), "--ci", m.group(5), m.group(6), "--threshold", m.group(1),
                       "--n-pos", m.group(4), "--n-neg", m.group(8), "--notes", notes]
    except Exception as e:
        err = getattr(e, "stderr", "") or str(e)
        args = base + ["--status", "failed", "--notes", err.strip().splitlines()[-1][:300] if err.strip() else repr(e)]
    rid = subprocess.run([PY, K / "runlog.py", "add"] + args, cwd=C, capture_output=True, text=True).stdout.strip()
    print(a.arm, d, rid, args[args.index("--status") + 1], args[args.index("--primary") + 1] if "--primary" in args else "",
          args[-1][:160], flush=True)
