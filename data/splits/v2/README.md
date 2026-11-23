# splits/v2 (built 2026-11-20)

Built by `GenIA-code/02_labeling/relation_v2/build_splits_v2.py` (seed 20261120). `splits/v1/` is unchanged and still
frozen; v2 is built **inside** v1, so no image changes side (the script asserts this).

| File | What | Used for |
|---|---|---|
| `train_v2.txt` | v1 train relation images (relabel) + new train images: every archikhu / sparks / platform / climb image with ≥ 1 person, + 300 random rf_harness_main images with people | Fitting the judge |
| `val_v2.txt` | Same recipe on the v1 val split (100 rf_harness_main) | Threshold and winner selection |
| `cs10k_dev.txt` | 150 CS10k images, not in the v1 queue, not in a duplicate cluster with the 159 CS10k test images, pHash Hamming > 10 to all of them, stratified by the same caption categories | Hard-negative false alarms for the threshold |
| `test_v1.txt` | The 336 real test photos, same selection as `07_ablation/evaluate.py` (143 indist + 159 CS10k + 34 heldout) | Final report only, once per decision |
| `manifest.csv` | One row per image: split_v2, split_v1, source, task_kind (relabel_v1 / new / frozen_test), selected_by, reason, synthetic, prefill people count, contrast group, min Hamming to test | Provenance |
| `build_report.md` | Counts per set and source; cs10k_dev quotas and exclusions | Audit |
| `SHA256SUMS` | Checksums of the five files above | `shasum -a 256 -c SHA256SUMS` |

**Status: FROZEN (2026-11-20, approved by Luciano).** The `FROZEN` file makes the builder refuse to run; changes go to splits/v3.
`calibration.txt` (INT8 calibration from train_v2 / val_v2, never test) is added at the quantization step.

Known limits: CS10k has no site ids, so cs10k_dev vs test is separated by image and near-duplicate only, not by site.
rf_scaffold_mobile is not added (video near-duplicates). Synthetic ESCO images are in train_v2 / val_v2 but flagged
(`synthetic = 1`); experiments keep `real_only` and `synthetic_fixed` as separate series.
