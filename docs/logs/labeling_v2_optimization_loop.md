---
title: Labeling v2 - optimization loop
created: 2026-11-20
tags: [genia, labeling, v2, control, loop]
---
# The labeling optimization loop (v2)

**Asked by Luciano, 2026-11-20.** Plan: [Labeling v2 and controlled experiment setup](labeling_v2_and_controlled_experiment_setup.md). Files: Labeling v2 - file map.

**In one sentence:** we improve the labels in rounds, and every round is judged the same way (same recipe, same seeds, same threshold rule, same decision rule, on validation only), so a better score means better labels and not luck, a different cutoff, or a test set we tuned to.

```
   ┌──────────────────────────────────────────────────────────────┐
   │ 1 Label  →  2 Train (fixed recipe, seeds 0-4)                 │
   │      ↑          ↓                                             │
   │ 6 Change    3 Threshold on val (FA = 1.5%, 5%)                │
   │   guidelines    ↓                                             │
   │   / labels  4 Control metrics + decision rule (val only)      │
   │      ↑          ↓                                             │
   │      └──── 5 Error analysis (train/val only)                  │
   └──────────────────────────────────────────────────────────────┘
        exit: rule says "stop" → 7 Test once (336 photos) → report
```

## The steps
| # | Step | What happens | Control that keeps it honest | Output |
|---|---|---|---|---|
| 1 | **Label** | Label a batch with the current guideline version (Claude first pass → Codex review → blind human check of ~50 people; 15% double-labeled) | Every image records `selected_by` (random / active / contrast / legacy_v1). Missing stays **unannotated**, never "no" | Export in `labels/relations/v2/`, guideline version tag |
| 2 | **Train** | Fit the judge (detector fixed, detections cached) on train_v2 | **One recipe** in `config.yaml`, written before the run. 5 draws, seeds `[0,1,2,3,4]`. Same tuning budget for every arm | Probabilities per val person; one RUNLOG row per draw |
| 3 | **Threshold** | Choose the cutoff where false alarms on val + cs10k_dev "not on" people = 1.5% (and 5%) | **One rule, val only.** Never chosen on test | Cutoff + recall at that cutoff |
| 4 | **Evaluate** | Primary: on-scaffold recall at FA 1.5%, end-to-end, with counts and bootstrap 95% CI. Plus C1 coverage, C2 κ, C3 balance, C4 learning curve, C5 error split | **Pre-written decision rule:** a change is kept only if it beats the baseline by ≥ 0.05 and the paired CI excludes 0. Ties → simpler | Kept / not kept, in RUNLOG and Ablation log §11 |
| 5 | **Error analysis** | Look at misses and false alarms: which `support_surface`, `proximity`, `feet_visible`, source; detector miss vs judge miss (C5) | **Train and val errors only.** Test errors are never looked at during the loop | A short list of error types with counts |
| 6 | **Change** | Either fix the guideline (new rule, clearer edge case → version bump), relabel the affected people, or choose what to label next (hard negatives, positives, contrast pairs) | Each guideline change gets a version and a reason. Relabeling is applied to train **and** val alike. A change in what to label is a new arm with an equal budget (R+N⁻ vs R+N⁺ …) | New guideline version, new batch spec → back to 1 |
| 7 | **Test once** | When the rule stops improving (or the budget ends), run the single chosen winner on test_v1, each subset separately (143 indist, 159 CS10k, 34 heldout) | **Opened once.** No changes after looking. The result is reported whatever it is | Final numbers |

## When the loop stops
- **C4 learning curve flat and good** → enough labels; stop and test.
- **Flat and poor** → the problem is representation/context (go to the context study C0–C3), not more labels.
- **Still rising** → label another batch of the type that helped (step 6).
- **κ < 0.6 on a field** → that field is not used as a target until the guideline is fixed.
- **No arm passes the rule twice in a row** → stop and test the baseline.

## What we never do
- Choose a threshold, a policy or which photos to label from **test** results.
- Compare arms trained with different recipes, seeds or thresholds.
- Drop failed runs from the log.
- Mix the `real_only` and `synthetic_fixed` series in one comparison.
- Train or label without Luciano's OK (training stays local on the Mac, lid open).

## Note on what already happened
The v1 exploration in [Ablation log](ablation_log.md) §10 looked at test results while choosing. That is reported openly. From v2 on, every choice is made on val.

## Interview line
"I treated label improvement as an optimization loop and controlled it like one: fixed recipe and seeds, a threshold rule on validation, a decision rule written before the runs, error analysis on train and val only, and the test opened once at the end."
