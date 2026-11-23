"""Study C2_structure driver: every arm x draw -> judge.py run -> threshold.py -> runlog.py add (ok or failed). No test data."""
import re, subprocess, sys, json
from pathlib import Path
E = Path(__file__).parent; C = E.parent.parent; PY = str(C / ".venv-train/bin/python"); K = C / "08_controlled"
CFG = "experiments/20261121_C2_structure/config.yaml"; STUDY = "C2_structure"
for arm in sys.argv[1:] or ["C1", "C1G", "C0", "C0G"]:
    for d in range(5):
        base = ["--study", STUDY, "--arm", arm, "--draw", str(d), "--config", CFG, "--label-version", "v2",
                "--split-version", "v2", "--series", "real_only", "--fa-budget", "0.015"]
        try:
            subprocess.run([PY, K / "judge.py", "run", "--config", CFG, "--arm", arm, "--seed", str(d), "--outdir", E],
                           cwd=C, check=True, capture_output=True, text=True)
            out = subprocess.run([PY, K / "threshold.py", E / f"scores_{arm}_d{d}.csv"], cwd=C, check=True, capture_output=True, text=True).stdout
            (E / f"threshold_{arm}_d{d}.txt").write_text(out)
            m = re.search(r"FA budget 1.5%: threshold (\S+) \| recall (\S+) \((\d+) of (\d+)\) 95% CI \[(\S+), (\S+)\] \| false alarms (\d+) of (\d+)", out)
            m5 = re.search(r"FA budget 5.0%: threshold \S+ \| recall (\S+) \((\d+) of \d+\) 95% CI \[(\S+), (\S+)\]", out)
            x = json.load(open(E / f"extra_{arm}_d{d}.json"))
            notes = (f"FA {m.group(7)} of {m.group(8)}; recall@5% {m5.group(1)} ({m5.group(2)} of {m.group(4)}) CI [{m5.group(3)}, {m5.group(4)}]; "
                     f"cond AP {x['conditional_ap']}; helmet bal {x['helmet_bal_acc']}; harness bal {x['harness_bal_acc']}; "
                     f"train imgs {x['train_images']} people {x['train_people_on_task']} pos {x['train_positives']}")
            args = base + ["--status", "ok", "--primary", m.group(2), "--ci", m.group(5), m.group(6), "--threshold", m.group(1),
                           "--n-pos", m.group(4), "--n-neg", m.group(8), "--notes", notes]
        except Exception as e:
            err = getattr(e, "stderr", "") or str(e)
            args = base + ["--status", "failed", "--notes", err.strip().splitlines()[-1][:300] if err.strip() else repr(e)]
        rid = subprocess.run([PY, K / "runlog.py", "add"] + args, cwd=C, capture_output=True, text=True).stdout.strip()
        print(arm, d, rid, args[args.index("--status") + 1], args[args.index("--primary") + 1] if "--primary" in args else "", flush=True)
