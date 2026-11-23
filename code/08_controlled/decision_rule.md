# Decision rule for controlled experiments (written 2026-11-20, before any v2 run)

This file is fixed **before** running. Changing it after a run counts as a new study and must be logged in RUNLOG.csv
(status `rule_change`, with the reason). Source: vault `Labeling/Labeling v2 and controlled experiment setup.md` §3.4.

## 1. Data
- **Fit** on `splits/v2/train_v2.txt` only.
- **Select** (threshold and winner) on `val_v2` + `cs10k_dev` only.
- **Test** on `test_v1` (336 photos, v1 labels): once, after the decision, each subset reported separately
  (143 test_indist, 159 test_external/CS10k, 34 test_heldout). No change after looking.
- `real_only` and `synthetic_fixed` are two separate series; never compared with each other inside one study.

## 2. Fixed recipe
- One `config.yaml` per study (from `config_template.yaml`), committed before the first run. Its SHA256 (first 12 chars)
  is written on every RUNLOG row. All arms in a study share it except the one field the study varies (`arm`).
- Seeds / draws: `[0, 1, 2, 3, 4]` for every arm. Any tuning gets the **same budget** for every arm, on val only.

## 3. Threshold rule (level 2)
- The judge outputs a probability. The cutoff is the **lowest score threshold whose false-alarm rate on the
  `not_on` people of val_v2 + cs10k_dev is ≤ 1.5%** (primary) and ≤ 5% (secondary).
- `unclear` and `unannotated` people are excluded from both recall and false alarms.
- End-to-end: an on-scaffold person the detector missed counts as a miss (score = 0).

## 4. Primary measure
- **On-scaffold recall at val false-alarm 1.5%**, end-to-end (detector + judge), on val_v2 (+ cs10k_dev negatives).
- Always reported with: counts ("9 of 620"), recall at 5%, conditional AP with true boxes, helmet and harness balanced
  accuracy, bootstrap 95% CI resampled **by image** (by sequence where a sequence id exists), 2,000 resamples,
  with the threshold re-chosen inside each resample.

## 5. Decision (level 3)
- An arm **replaces the baseline only if** its primary measure (mean over the 5 draws) beats the baseline by
  **≥ 0.05** **and** the paired bootstrap 95% CI of the difference excludes 0.
- Otherwise keep the baseline. **Ties go to the simpler arm** (fewer labels, fewer inputs, no synthetic).
- The decision is written to RUNLOG.csv (`status = decision`) **before** any test run. `runlog.py check-test`
  refuses a test run without it, and refuses a second test run for the same study.

## 6. Labeling loop (level 4)
- Guideline changes and the choice of what to label next come from **train/val errors only**.
- Every guideline change bumps its version (GUIDELINES.md §9) with the reason.
- Every labeled image records `selected_by`. Added-data arms use equal budgets (R+N⁻, R+N⁺, R+N±, R+2N).
- Fields with κ < 0.6 are not used as training targets.

## 7. Stop rule
- Stop and test when the C4 learning curve is flat (last doubling < 0.02 gain), or when no arm passes §5 twice in a row,
  or when the labeling budget ends.

## 8. What is logged
Every run, including failed and abandoned ones, gets one RUNLOG.csv row. Nothing is deleted from RUNLOG.csv.
