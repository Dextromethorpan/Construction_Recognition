# 08_controlled: controlled experiments (labeling v2)

`07_ablation/` stays as the record of the v1 ablation (its `evaluate.py` is reused for the final test). This folder holds
the control machinery for every experiment from v2 on. Vault notes: `Labeling/Labeling v2 and controlled experiment setup.md`,
`Labeling/Labeling v2 - optimization loop.md`, `Labeling/Labeling v2 - file map.md`.

| File | Status | Purpose |
|---|---|---|
| `decision_rule.md` | fixed 2026-11-20 | Data use, fixed recipe, threshold rule, primary measure, ≥ 0.05 + CI rule, test once |
| `config_template.yaml` | ready | Copy to `experiments/<date>_<study>/config.yaml` before the first run of a study |
| `runlog.py` | ready | Append-only `experiments/RUNLOG.csv`; `decide`; `check-test` blocks a test run without a decision |
| `threshold.py` | ready | Cutoff at val FA 1.5% / 5%, recall with counts and bootstrap CI; paired arm vs baseline verdict. Refuses test rows |
| `control_metrics.py` | built 2026-11-21 | C1 coverage, C2 κ (wraps `02_labeling/relation_v2/kappa_v2.py`), C3 balance, C4 learning curve, C5 error split |
| `judge.py` | built 2026-11-21 | Judge with probability output and C0–C3 inputs, cached detections/features (ideas from 07 `design_B.py`, not an edit). `prepare` then `run --arm --seed --outdir [--train-frac]` |
| `compare.py` | built 2026-11-21 | Study-level paired verdict: mean over draws of arm − baseline, one image resample shared by all draws |

## One study, step by step
1. `mkdir experiments/<YYYYMMDD>_<study>` and fill `config.yaml` from the template. Commit it (its hash is logged).
2. For each arm and seed 0–4: run the judge, write `scores_<arm>_d<seed>.csv` for val_v2 + cs10k_dev
   (columns: image_id, person_id, set, on_scaffold, score, group_id).
3. `python threshold.py scores_<arm>_d<seed>.csv` and log it:
   `python runlog.py add --study <s> --arm <arm> --draw <seed> --config experiments/<...>/config.yaml --label-version v1 --status ok --primary <recall> --ci <lo> <hi> --fa-budget 0.015 --threshold <t> --n-pos <n> --n-neg <n>`
   Failed or abandoned runs: same command with `--status failed|abandoned --notes "why"`.
4. Compare: `python threshold.py scores_<arm>_d<seed>.csv --baseline scores_baseline_d<seed>.csv` per seed;
   the primary is the mean over the 5 seeds, the verdict follows `decision_rule.md` §5.
5. `python runlog.py decide --study <s> --winner <arm> --notes "<numbers>"`.
6. Only then: `python runlog.py check-test --study <s>` → run `07_ablation/evaluate.py` style scoring on test_v1 once,
   and log it with `--status test`. Summarise in vault `Training/Ablation log.md` §11.

Training stays local on the Mac (MPS), lid open. No run starts without Luciano's OK.
