# Study 1: context C0-C3 (2026-11-21). Config written before the first run: config.yaml. Decision: **C1** (RUNLOG r00022).

Selection set: val_v2 + cs10k_dev, real only: 118 on-scaffold, 878 not_on people. End-to-end (detector misses score 0).
5 draws = random 80% of train_v2 images each (564 images, ~1,490 people, ~199 on). Every run in RUNLOG r00001-r00020.

| Arm | Recall @FA 1.5% (mean of 5) | hits / 118 per draw | 95% CI (mean of draw CIs) | FA count | Recall @FA 5% | hits | 95% CI | cond AP | helmet bal acc | harness bal acc | val_v2 recall @1.5% | cs10k_dev recall @1.5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.454 | 55,49,54,54,56 | [0.299, 0.586] | 13 of 878 | 0.540 | 61,66,64,64,64 | [0.425, 0.663] | 0.643 | 0.807 | 0.812 | 45/65, 40/65, 46/65, 44/65, 46/65 | 10/53, 9/53, 8/53, 10/53, 10/53 |
| C1 | 0.534 | 63,65,62,64,61 | [0.408, 0.658] | 13 of 878 | 0.617 | 72,71,72,75,74 | [0.505, 0.728] | 0.783 | 0.770 | 0.757 | 49/65, 49/65, 47/65, 47/65, 45/65 | 14/53, 16/53, 15/53, 17/53, 16/53 |
| C2 (invalid: no platform boxes) | 0.454 | 55,49,54,54,56 | [0.300, 0.586] | 13 of 878 | 0.540 | 61,66,64,64,64 | [0.425, 0.663] | 0.643 | 0.806 | 0.812 | 45/65, 40/65, 46/65, 44/65, 46/65 | 10/53, 9/53, 8/53, 10/53, 10/53 |
| C3 | 0.487 | 57,58,54,60,58 | [0.325, 0.615] | 13 of 878 | 0.590 | 69,69,71,68,71 | [0.479, 0.712] | 0.705 | 0.835 | 0.828 | 48/65, 48/65, 48/65, 49/65, 50/65 | 9/53, 10/53, 6/53, 11/53, 8/53 |

CI per arm = mean of the per-draw bootstrap CIs (threshold.py, 2000 image resamples, threshold re-chosen inside).

## Paired vs C0 (compare.py: mean over 5 draws, one image resample applied to all draws, 2000 resamples)
| Arm | FA 1.5% diff [95% CI] | FA 5% diff [95% CI] | Verdict (§5) |
|---|---|---|---|
| C1 | **+0.080 [+0.014, +0.187]** | +0.076 [+0.026, +0.122] | **REPLACE** |
| C2 | +0.000 (identical to C0) | +0.000 | invalid run, see below |
| C3 | +0.032 [-0.044, +0.092] | +0.049 [+0.011, +0.105] | KEEP |
Conditional (labeled boxes): C1 +0.253 [+0.137, +0.319], C3 +0.110 [+0.023, +0.198] at FA 1.5% (500 resamples).

## Caveats
- C1's CI lower bound is close to 0 and only 1 of 5 draws passes on its own; the gain comes mostly from cs10k_dev (≈ 16/53 vs 9/53).
- C1 lowers helmet (0.770 vs 0.807) and harness (0.757 vs 0.812) balanced accuracy: keep the C0 crop for the PPE heads.
- **C2 is untested**: the fixed detector emits no scaffold_platform boxes at conf 0.25, so its geometry features were
  constant (RUNLOG r00021, abandoned). Re-testing C2 is a new study (different platform source).
- After C1, detector misses are 64% of remaining misses at FA 1.5% (35 of 55): detection on small CS10k people is the
  next bottleneck, not the judge.
- Test (test_v1) NOT run.
