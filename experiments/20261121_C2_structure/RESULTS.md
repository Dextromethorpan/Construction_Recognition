# Study C2_structure (2026-11-21): support geometry from scaffold_structure boxes. Decision: **keep C1** (RUNLOG r00043).

Config written before the first run (config.yaml). Same recipe as context_C0_C3; detections and C0/C1 features are byte-identical
to that study. Geometry source: the detector's scaffold_structure boxes (class 5, conf >= 0.25), same for train and val.
Runs r00023-r00042. Selection set val_v2 + cs10k_dev (real): 118 on, 878 not_on. End-to-end. Test NOT run.

## Feature check (before running): not constant
| People (labeled boxes) | n | structure box in image | feet within 1 person-height | feet inside a structure box |
|---|---|---|---|---|
| train on | 245 | 215 (88%) | 215 (88%) | 207 (84%) |
| train not_on | 1,638 | 183 (11%) | 155 (9%) | 95 (6%) |
| selection on | 118 | 82 (69%) | 81 (69%) | 78 (66%) |
| selection not_on | 878 | 79 (9%) | 64 (7%) | 44 (5%) |

## Results (mean of 5 draws)
| Arm | Recall @FA 1.5% | hits of 118 | mean draw CI | FA | Recall @FA 5% | hits | cond AP | helmet | harness | val_v2 hits/65 | cs10k_dev hits/53 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 (baseline) | 0.534 | 63,65,62,64,61 | [0.408, 0.658] | 13 of 878 | 0.617 | 72,71,72,75,74 | 0.783 | 0.770 | 0.757 | 47.4 | 15.6 |
| C1G | 0.546 | 63,65,63,67,64 | [0.417, 0.665] | 13 of 878 | 0.621 | 72,71,73,75,75 | 0.790 | 0.769 | 0.756 | 48.8 | 15.6 |
| C0 | 0.454 | 55,49,54,54,56 | [0.299, 0.586] | 13 of 878 | 0.540 | 61,66,64,64,64 | 0.643 | 0.807 | 0.812 | 44.2 | 9.4 |
| C0G | 0.493 | 56,59,59,60,57 | [0.336, 0.612] | 13 of 878 | 0.553 | 65,67,65,66,63 | 0.658 | 0.791 | 0.815 | 47.0 | 11.2 |

## Paired (compare.py, 2000 image resamples shared across draws)
| Comparison | FA 1.5% diff [95% CI] | FA 5% diff [95% CI] | Verdict |
|---|---|---|---|
| **C1G vs C1 (primary)** | +0.012 [-0.011, +0.033] | +0.003 [-0.003, +0.023] | **KEEP C1** (< 0.05) |
| C0G vs C0 (info) | +0.039 [+0.002, +0.080] | +0.012 [-0.002, +0.026] | below 0.05 |
Conditional (labeled boxes) C1G vs C1: -0.010 [-0.032, +0.048].

Reading: the structure geometry separates the classes strongly on its own, but the bigger C1 crop already sees the
scaffold, so adding it to C1 gives almost nothing. It helps the small C0 crop a little (+0.04), still under the 0.05 bar.
Remaining misses are mostly detector misses (35 of 118 on-scaffold people get no box), which no judge input can fix.
