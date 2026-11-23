# Ablation log (labeling policies)

Running record of the labeling-policy ablation. Started 2026-11-19.

## 1. Goal
Compare labeling policies (unclear handling, harness source, Claude vs Codex labels, synthetic ESCO in or out) with the same split, model, hyperparameters and seed. Score them on real photos for helmet and harness accuracy, on-scaffold F1 and hard negatives.

## 2. Labels frozen (2026-11-19 16:48)
- `~/GenIA-data/labels/relations/v1/frozen_v1/` is read-only and checked with `shasum -a 256 -c SHA256SUMS`.
- `codex_reviewed.json` is the final Codex-reviewed export (`export_20261119_145823.json`).
- `claude_specs.tar` holds Claude's originals, `codex_specs.tar` holds the Codex specs and reviews, and the queue, synthetic list and task logs are copied alongside.

## 3. Finding: the variants barely touch the detector's training data
Relation labels (support, helmet/harness state) exist only for the 754 queue images. Most of them are test images.

| Split | Images | Real people (on / next to / unclear) | Synthetic people |
|---|---|---|---|
| train | 168 (113 synthetic) | 88 (63 / 18 / 7) | 154 |
| val | 50 (35 synthetic) | 39 | 54 |
| test_indist | 334 (191 synthetic) | 277 | 251 |
| test_external (CS10k) | 159 | 745 (42 on) | 0 |
| test_heldout | 43 | 62 | 0 |

The detector itself trains on 13,597 images whose boxes come from the source datasets. A full YOLO retrain per variant would therefore change only about 1% of the data, and the results would be mostly noise.

## 4. Design options (decided 2026-11-19 16:05)
- **Option A:** retrain YOLO per variant, as originally planned (2 to 6 h on the Mac).
- **Option B (recommended):** train the detector once (about 1 h on the M5), then run each label policy as a small per-person second stage (on-scaffold, helmet state, harness state). Score it with 5-fold cross-validation grouped by source over the real labeled images (about 590 people), and keep CS10k as the untouched hard-negative test. Every variant runs in minutes.
- Speed notes: the Mac is an M5 with an 8-core GPU and 16 GB. Running two jobs at once isn't worth it. Kaggle's 2x T4 would be about 2x faster for screening, but it needs a sign-in and an upload.
- Installing PyTorch and Ultralytics into `~/GenIA-code/.venv-train` needs Luciano's OK.

**Decision:** Luciano chose to run on the Mac, approved installing PyTorch and Ultralytics, and asked for **A first, then B**, so the two designs can be compared.

## 5. Setup
- **Environment:** `.venv-train` has PyTorch 2.15 and Ultralytics 8.4, and MPS works. The lock file is `~/GenIA-code/envs/train-requirements.lock.txt`.
- **Measured speed (yolo11n, M5 GPU):** 4.4 min per epoch at 512 px and 7.1 min at 640 px on the full 13.4k training set. That's 3.2 and 2.0 batches/s, batch 16.
- **Code:** everything is in `~/GenIA-code/07_ablation/`: `build_variants.py`, `finetune.py`, `evaluate.py`, `run_all.sh` and `design_B.py`.

## 6. Variants (the same in A and B)
| ID | Change vs base |
|---|---|
| V0_base | Codex labels. `unclear` people dropped (in A they're masked out with a grey fill). Harness from harness boxes |
| V1_unclear_class | `unclear` kept as its own class (a predicted unclear counts as not on scaffold) |
| V2_harness_field | Harness from the per-person `harness_state` field instead of boxes |
| V3_claude | Claude's pre-review labels (`export_20261118_220544.json`) instead of Codex's. Claude drew almost no helmet or harness boxes, so in A those come from the source datasets |
| V4_no_synth | No synthetic ESCO images |
| V5_unclear_neg | `unclear` counted as not on scaffold |
Evaluation always uses the frozen Codex labels, on real photos only.

## 7. Design A as run (fast version)
- **Shared base:** yolo11n at 512 px, 15 epochs, seed 0, trained on all scheme-B data. The classes `person_on_scaffold`, `person_on_unclear` and `person_harness_worn` are reserved, so every variant fine-tunes from the same checkpoint with the same head.
- **Per variant:** fine-tune for 10 epochs (AdamW, lr 0.001, seed 0) on the variant's 168 relation-labeled training images plus a fixed replay sample of 2,000 PPE images (hardhat_wkrs, shwd, rf_harness_fypcv). Validation uses the 50 relation-labeled val images plus 300 replay images. Scaffold sources are left out of the replay because their people carry no on-scaffold label.
- **Scoring:** 336 real test photos with 1,078 people (test_indist 143, CS10k 159, heldout 34). Each person is matched to the predictions at IoU 0.5 with confidence 0.25. Metrics: on-scaffold P/R/F1, where `person_on_scaffold` boxes that hit no labeled person count as false positives; hard-negative false-positive rate on CS10k `none` people; helmet and harness balanced accuracy.

## 8. Design B as run
- **Model:** a yolo11n-cls embedding of each person crop with context (box widened 2x, heightened 1.5x), plus box geometry, fed to a class-balanced logistic regression.
- **Cross-validation:** 5 folds over the real dev people (train, val, test_indist and heldout; n = 436 for on-scaffold). Folds are grouped by image, and by whole source for the video sources. Synthetic people are used only for training.
- **Holdout:** CS10k, with the model trained on the whole dev pool.
- **Inputs:** GT person boxes are used, so B measures the label policy alone, not person detection. Bootstrap 95% CIs are included.
- Results are in `~/GenIA-data/runs/ablation_B/scores.json`.

## 9. Results (2026-11-19, A finished 19:51 CET)

**Design A** scores every variant on 336 real test photos with 1,078 people. On-scaffold F1 has 191 people on a scaffold to find. Helmet and harness are balanced accuracy. Hard-neg FP is the share of the 620 CS10k people who are not on a scaffold but get flagged as on one.

| Variant | On F1 | P | R | Helmet | Harness | Hard-neg FP | Person recall |
|---|---|---|---|---|---|---|---|
| base only (no fine-tune) | 0.00 | – | 0.00 | 0.65 | 0.57 | 0% | 0.25 |
| V0_base | **0.57** | 0.75 | 0.46 | 0.68 | 0.70 | 0.6% | 0.16 |
| V1_unclear_class | **0.57** | 0.71 | 0.47 | 0.68 | 0.69 | 0.6% | 0.18 |
| V2_harness_field | 0.55 | 0.77 | 0.42 | 0.70 | **0.75** | 0.3% | 0.16 |
| V3_claude | 0.55 | 0.75 | 0.43 | 0.63 | 0.55 | 0.6% | 0.18 |
| V4_no_synth | 0.56 | 0.79 | 0.43 | 0.70 | 0.67 | 0.0% | 0.18 |
| V5_unclear_neg | 0.55 | 0.74 | 0.44 | 0.69 | 0.71 | 0.3% | 0.17 |

**Design B** scores each variant with 5-fold CV on real dev people (n = 436). CIs come from a person-level bootstrap. Hard-neg FP is on CS10k with GT person boxes.

| Variant | On F1 (95% CI) | Hard-neg FP |
|---|---|---|
| V0_base | **0.83** (0.79–0.86) | 24% |
| V1_unclear_class | 0.78 | 24% |
| V2_harness_field | 0.83 | 24% |
| V3_claude | 0.79 | 18% |
| V4_no_synth | 0.79 | 26% |
| V5_unclear_neg | 0.79 | 35% |

**Reading**
- **On-scaffold:** in A, every variant lands within 0.02 F1 (0.55 to 0.57). With 191 positives that's inside the noise, as predicted, because the variants change only about 1% of the detector's data. B separates the variants a little more, and V0 (drop unclear) is best or tied in both designs.
- **Harness:** taking harness from the per-person field (V2) is the clearest win in A, at 0.75 against 0.70.
- **Codex review:** Claude's pre-review labels (V3) are the worst for helmet and harness in A, because Claude drew almost no PPE boxes. This supports the Codex review.
- **Unclear handling:** counting unclear as "not on" (V5) gives no gain in A and the worst hard-negative rate in B (35%). Keeping unclear as its own class (V1) doesn't help either.
- **Synthetic images:** dropping them (V4) removes the last hard-negative false alarms in A, with no loss in F1.
- **Bottleneck:** person detection. The detector finds only 16 to 25% of the people, because person labels in the source datasets are sparse, and that caps A's recall. The next win is exhaustive person labels or a COCO-pretrained person detector, not label policy.
- **The two designs' numbers aren't directly comparable.** B starts from GT person boxes, so its F1 is higher. A detects and classifies end to end, so it has fewer false alarms but much lower recall.

**Recommended policy:** V0 + V2. That means drop unclear, take harness from the per-person field, use Codex-reviewed labels, and keep synthetic images (or drop them if hard negatives matter most).

## 10. Steps 1 and 2: person coverage and the two-stage pipeline (2026-11-19 evening, exploration)

These runs are exploration, chosen by hand after looking at test scores. The controlled runner (section 11, next) will redo the selection without touching test.

- **Step 1, complete the person labels:** COCO-pretrained yolo11m (conf ≥ 0.5) added person boxes that didn't overlap an existing one. It added 34,459 boxes in train and 8,219 in val, giving `build/B_persons_v1` (`07_ablation/pseudo_persons.py`).
- **Bug (fixed):** the first base retrain read the old labels. The `images/` folder was a symlink, and Ultralytics resolved it back to `B_objects_platform_v1`, so that base was an exact copy of the old one. It's kept as `runs/detector/_invalid_base2_trained_on_old_labels`. The fine-tune did use the completed labels. The fix is per-file symlinks, and the rerun is `run_step1b.sh`.
- **Step 2, two stages:** the detector finds people, then the per-person classifier (design B features) judges on-scaffold, helmet and harness (`07_ablation/combined.py`). The classifier is trained only on the train and val relation images, and `--neg 400` adds 400 people from ground-level helmet photos as "not on scaffold" examples.

| Pipeline (336 real test photos) | People found | On P / R / F1 | False alarms CS10k | Helmet | Harness |
|---|---|---|---|---|---|
| First results V2 (section 9) | 16% | 0.77 / 0.42 / 0.55 | 0.3% | 0.70 | 0.75 |
| Step 1 partial: completed persons in the fine-tune only | 51% | 0.72 / 0.45 / 0.55 | 0.3% | 0.68 | 0.71 |
| Step 2: COCO yolo11m + classifier | 66% | 0.39 / 0.62 / 0.48 | 25% | 0.69* | 0.66* |
| Step 2: COCO yolo11m + classifier + 400 ground negatives | 66% | 0.49 / 0.56 / 0.52 | 15% | 0.69* | 0.66* |
| Step 2: own detector (step 1 partial) + classifier | 51% | 0.40 / 0.57 / 0.47 | 18% | 0.70* | 0.69* |

\* The two-stage pipeline's helmet and harness scores are measured only on the people it found.

**Reading:** completing the person labels triples person recall, but on-scaffold F1 stays at about 0.55, because the extra people found are mostly on the ground. The two-stage pipeline trades precision for recall: it catches more people on scaffolds and raises more false alarms. Its classifier is trained on only 261 people, 84% of them on a scaffold, so it leans towards "on". The pending results are the step 1 rerun with a properly retrained base.

**Step 1 rerun (base retrained on the completed person labels, finished 2026-11-20 12:14)**

The Mac slept overnight from epoch 11. Closing the lid overrides `caffeinate -i`, so keep the lid open for long runs.

| Pipeline | People found | On P / R / F1 | False alarms CS10k | Helmet | Harness |
|---|---|---|---|---|---|
| New base only (no fine-tune) | 56% | – | 0% | 0.65 | 0.59 |
| Step 1: new base + V2 fine-tune | 55% | 0.69 / 0.43 / 0.53 | 1.5% | 0.69 | 0.72 |
| Step 1 + 2: new base + V2 fine-tune + classifier (+400 negatives) | 55% | 0.49 / 0.53 / 0.51 | 12% | 0.68* | 0.69* |

**Conclusion of the exploration**
- Completing the person labels fixed person finding: 16% went to 55%, at no cost to helmet and harness.
- On-scaffold F1 did not move. It stayed between 0.51 and 0.57 in every setup. The bottleneck is now the on-scaffold judgement itself, which learns from only about 88 real training people (63 on a scaffold, 18 not). It is no longer person detection.
- The single-model pipeline is precise (P ≈ 0.7, false alarms ≤ 1.5%) but misses more than half the people on scaffolds. The two-stage pipeline catches more of them (R ≈ 0.53 to 0.62), with 12 to 25% false alarms. Choosing between them is a threshold and policy decision, and that choice belongs in the controlled runner's pre-written rule.
- **Next:** the controlled runner (config, rule written first, validation-only selection, every run logged, test once for the winner), and more real on-scaffold examples.

## 11. Controlled runs (v2 labels, validation-only selection)

### 11.1 Study 1: context C0–C3 (2026-11-21) → **C1 kept**
Labels `frozen_v2` (frozen 2026-11-21, 203 claude_only tasks flagged), series real_only, fixed detector `runs/ablation_v2b/V2_harness_field/weights/best.pt` (imgsz 512, conf 0.25), 5 draws of 80% of train_v2. Selection set val_v2 + cs10k_dev: 118 on, 878 not_on. Code `GenIA-code/08_controlled/` (judge.py, control_metrics.py, compare.py); files `GenIA-code/experiments/20261121_context_C0_C3/` (config, RESULTS.md); RUNLOG r00001–r00022.

| Arm | Recall at FA 1.5% (mean of 5) | Hits of 118 | Recall at FA 5% | Cond. AP | Paired diff vs C0 at 1.5% [95% CI] |
|---|---|---|---|---|---|
| C0 crop as today | 0.454 | 49–56 | 0.540 | 0.64 | — |
| C1 3×w, 2×h, aspect kept | **0.534** | 61–65 | 0.617 | 0.78 | **+0.080 [+0.014, +0.187] → replace** |
| C2 C0 + support geometry | (= C0) | — | — | — | invalid: detector gives no platform boxes |
| C3 3×3 spatial pooling | 0.487 | 54–60 | 0.590 | 0.71 | +0.032 [−0.044, +0.092] → keep C0 |

False alarms 13 of 878 at the 1.5% cutoff. C1 costs some helmet (0.77 vs 0.81) and harness (0.76 vs 0.81) balanced accuracy, so keep the C0 crop for the PPE heads. **Test not run.**

### 11.0 Study 0: free checks (2026-11-21)
`GenIA-code/experiments/20261121_study0_free_checks/FINDINGS.md`. Class balance has flipped: 6.7 negatives per positive in train_v2 (v1: 0.29), with 8% hard negatives at height. Error split for C0: 56% of on-scaffold misses are detector misses (CS10k: 23 of 53 on-people get no box). Judge AP with true boxes: 0.90 on val_v2 but 0.30 on cs10k_dev. The neg400 people are not at height, but about a third are not construction. About 7 of 30 fypcv replay people are at height. The fixed detector never predicts scaffold_platform at conf 0.25.

### 11.2 Study C2_structure: support geometry from scaffold_structure boxes (2026-11-21) → **keep C1**
Luciano's choice: the geometry comes from the detector's scaffold_structure boxes (conf 0.25), because it predicts no platforms. The recipe is the same as 11.1 and the baseline is C1. Files: `GenIA-code/experiments/20261121_C2_structure/` (config, RESULTS.md). RUNLOG r00023–r00043. The features are not constant: 84% of on-scaffold train people have their feet inside a structure box, against 6% of not_on people (selection set: 66% vs 5%).

| Arm | Recall at FA 1.5% (mean of 5) | Hits of 118 | Recall at FA 5% | Paired diff [95% CI] |
|---|---|---|---|---|
| C1 (baseline) | 0.534 | 61–65 | 0.617 | — |
| C1 + structure geometry | 0.546 | 63–67 | 0.621 | **+0.012 [−0.011, +0.033] vs C1 → keep C1** |
| C0 (info) | 0.454 | 49–56 | 0.540 | — |
| C0 + structure geometry (info) | 0.493 | 56–60 | 0.553 | +0.039 [+0.002, +0.080] vs C0 (under 0.05) |

The bigger C1 crop already sees the scaffold, so the geometry adds almost nothing on top of it. The judge stays **C1**. The remaining misses are mostly detector misses. **Test not run.**

### 11.3 Study D1_detector: which detector feeds the C1 judge (2026-11-21) → **keep D0**
Hypothesis (written before running): 35 of 118 on-scaffold people get no box, mostly small CS10k people, so more input pixels should box more of them and gain ≥ 0.05. Judge fixed at C1, same 5 draws. Each detector arm got one training seed (time budget). Files: `GenIA-code/experiments/20261121_D1_detector/` (config, run_arm.py, tiled_detect.py, train_arms.sh). RUNLOG r00044–r00059, r00064–r00074.

| Arm | What changed | Recall at FA 1.5% (mean of 5) | On-scaffold people boxed (of 118) | Spurious boxes | Paired diff vs D0 [95% CI] |
|---|---|---|---|---|---|
| **D0 (baseline)** | yolo11n, 512 px | **0.534** | 83 | 60 | — |
| D1_960 | fine-tune redone at 960 px | 0.573 | 91 | 61 | +0.039 [−0.024, +0.110] → keep D0 |
| D2 | yolo11s (base + fine-tune), 512 px | 0.522 | 84 | 50 | −0.012 [−0.072, +0.051] → keep D0 |
| D0 at 1280 px (info) | inference only, no retraining | 0.512 | 96 | 183 | −0.022 [−0.105, +0.054] |
| D0 tiled (info) | full image + 2×2 tiles, NMS | 0.586 | 97 | 483 | +0.053 [−0.015, +0.113] |

- The 1280 px fine-tune was abandoned (r00044): batch 8 used 10.4 GB and swapped on the 16 GB Mac, with about 2 days projected. It was replaced by 960 px before any scoring.
- Reading: higher resolution and tiling do box more people (up to 97 of 118). The extra boxes are mostly small people the judge still rejects, and both add many unlabeled spurious boxes, which the FA rule does not count. The bigger model does not help small people. No arm clears the bar. **D1_960 is the runner-up worth naming in the interview.**

### 11.4 Study H1_harness_crop: torso crop for the harness head (2026-11-21) → **keep the C0 crop**
Measured on labeled boxes, independent of the detector. Harness κ v2 = 0.66, so it is allowed as a target. n = 312 (68 worn). RUNLOG r00060–r00063. Files: `experiments/20261121_H1_harness_crop/`.

| Arm | Harness balanced acc. (mean of 5) | Harness AP | Diff vs H0 [95% CI] | Helmet balanced acc. |
|---|---|---|---|---|
| **H0 C0 crop (today)** | **0.812** | 0.790 | — | 0.807 |
| H1 torso crop | 0.773 | 0.742 | −0.040 [−0.076, +0.010] | 0.663 |
| H2 C0 + torso (info) | 0.825 | 0.836 | +0.013 [−0.008, +0.038] | 0.785 |

The hypothesis was rejected: the tight torso crop loses the shoulder and lanyard context.

### 11.5 Study Q1_quantize: FP16 / INT8 export of the chosen detector D0 (2026-11-21) → **CoreML INT8**
Rule (written before running): recommend the smallest variant whose recall drop is ≤ 0.02 and whose boxed-people drop is ≤ 2. Calibration used 200 train_v2 images only. Speed is the median over 100 val images, batch 1, on the M5, including pre-processing and NMS. The judge encoder stays FP32. Code: `GenIA-code/06_quantize/`, models in `GenIA-data/runs/quantize/D0/`. RUNLOG r00075–r00095.

| Variant | Runs on | Size (MB) | Median ms / image | Recall at FA 1.5% | Boxed (of 118) | Drop vs FP32 |
|---|---|---|---|---|---|---|
| PyTorch FP32 (reference) | MPS GPU | 5.49 | 9.3 | 0.534 | 83 | — |
| CoreML FP16 | Apple Neural Engine / CPU | 5.40 | 4.2 | 0.532 | 83 | 0.002 |
| **CoreML INT8 (8-bit weights)** | Apple Neural Engine / CPU | **2.86** | **4.1** | **0.534** | **83** | **0.000** |
| ONNX FP32 | CPU | 10.61 | 22.8 | 0.529 | 82 | 0.005 |
| ONNX INT8 (static, QDQ) | CPU | 3.19 | 18.7 | 0.508 | 82 | 0.026 (over budget) |

The .pt file is small because ultralytics stores its weights as FP16. Full INT8 (activations too) costs about 3 points of recall on CPU. Weight-only INT8 on CoreML is free.

### 11.6 Where the loop stands (2026-11-21 ~20:30 CET)
Stop rule §7 is met: no arm passed the bar in three studies in a row (C2_structure, H1_harness_crop, D1_detector). **Final pipeline:** detector D0 (yolo11n, 512 px), then the on-scaffold judge C1, plus helmet and harness heads on the C0 crop. Deployment build: CoreML INT8. Cutoff at val FA 1.5%. **The test set has not been opened.**

### 11.7 One-time test (2026-11-21 20:05 CET, Luciano's OK) → see [Final results](final_results.md)
Decision row r00096 written before the test, test rows r00097. All real photos: recall 0.517 [0.433, 0.608] at the val cutoff, with a false-alarm rate of 5.4% (validation: 1.5%). In-distribution 0.805, CS10k 0.176, held-out 0.072. CoreML INT8: 0.521.

## 12. Post-test round (2026-11-21 evening, VALIDATION ONLY, test already used)
Luciano asked to keep training through Sunday. The rule and the bar are unchanged. Nothing here has a test number, and nothing here may be presented as one. CS10k train images were allowed for this round only (Luciano, 20:47 CET), excluding test, cs10k_dev and near-duplicates.

| Study | Change | Recall at FA 1.5% | Boxed (of 118) | Diff vs D0 [95% CI] | Verdict |
|---|---|---|---|---|---|
| J1_judge_finetune | Crop classifier fine-tuned end-to-end | 0.515 | 83 | −0.019 [−0.082, +0.047] | keep |
| J1 (info) | MLP on frozen features | 0.520 | 83 | −0.014 | keep |
| T1_threshold | Conservative cutoff (bootstrap 90th percentile) | 0.496 | 83 | −0.041 | keep (see below) |
| K1_cs10k_persons | 960 px + 2,000 CS10k images with yolo11m person boxes | 0.510 | 86 | −0.024 [−0.093, +0.063] | keep |
| S1_operating_point | D0, conf 0.10 | 0.539 | 90 | +0.005 | keep |
| S1 | 960 px, conf 0.10 | 0.576 | 96 | +0.042 [−0.023, +0.102] | keep |
| S1 | **960 px, conf 0.10, NMS 0.8** | **0.585** | **98** | **+0.051 [−0.014, +0.112]** | keep (CI includes 0) |

What we learned:
- **The judge is not the limit.** A fine-tuned or MLP judge does no better end-to-end. With perfect boxes the linear judge already has AP 0.96 on val_v2 and 0.48 on cs10k_dev.
- **The test false-alarm jump is a domain shift.** Over 500 val half-splits the point cutoff keeps realised FA at a median of 1.58% (p90 3.1%). Threshold noise explains little of the test's 5.4%.
- **The pseudo labels taught the wrong thing.** yolo11m at conf 0.5 finds only 2 of the 27 people our 960 px detector misses, so its CS10k labels mark those people as background.
- **What the missed people look like:** a median of 11% of image height (not tiny), in crowded groups, behind scaffold tubes, in low contrast. Lower confidence plus a looser NMS boxes up to 98 of 118.
- **The study is underpowered.** Every 960 px variant gains +0.04 to +0.05, but with 118 positives the CI half-width is about 0.06. Confirming a 0.05 gain needs about 250–300 on-scaffold people in the selection set.

### 12.1 HL1_helmet_crop (2026-11-21 23:45 CET, validation only) → keep C0; C0 + head crop flagged
Helmet κ is 0.89. n = 1,013 labeled people (768 worn, 245 not worn). RUNLOG r00140–r00143.

| Arm | Helmet balanced acc. | Not-worn caught (of 245) | Worn correct | Diff vs C0 [95% CI] |
|---|---|---|---|---|
| **C0 crop (today)** | 0.807 | 76.7% | 84.7% | — |
| Head crop (primary arm) | 0.821 | 74.9% | 89.3% | +0.015 [−0.024, +0.050] → keep C0 |
| C0 + head crop together (declared info-only before running) | **0.858** | **79.3%** | **92.3%** | **+0.051 [+0.026, +0.077]** |

The combined input clears the bar. However, it was declared info-only before the run, so adopting it would be a post-hoc choice. It is recorded as the recommended next change, not as a confirmed winner. The head crop alone hurts harness (−0.080), so harness keeps C0.

### 12.2 V3_confirm: the two candidates re-checked on NEW photos (2026-11-22 night) → helmet change confirmed, detector change not
Luciano chose option A. The rules were written before any image was selected (`experiments/20261122_V3_confirm/config.yaml`), with one amendment after 8 images: images are now selected by caption ("on/climbing a scaffold") because positives were rare. The set: 120 new CS10k photos, not near any test, cs10k_dev or val image, with 504 people (200 on scaffold, 235 not on, 69 unclear). Labels are **Claude only** (no second annotator). Person boxes were prefilled from two detectors and corrected by hand. Files: `GenIA-data/labels/relations/v2x/val_extra_people.csv`, `GenIA-code/02_labeling/val_extra/labels.txt`. RUNLOG r00144–r00155.

| Candidate | Baseline | Candidate | Diff [95% CI] | Verdict |
|---|---|---|---|---|
| Detector 960 px + conf 0.10 + NMS 0.8: recall at FA 1.5% | 0.037 | 0.070 | +0.033 [−0.089, +0.057] | **not confirmed**, keep D0 |
| Same: on-scaffold people boxed (of 200) | 106 | **139** | +33 | (info) |
| Helmet head C0 + head crop: balanced accuracy | 0.630 | **0.763** | **+0.133 [+0.017, +0.280]** | **confirmed** |
| Same: no-helmet people caught (of 16) | 40% | 59% | | (info) |

Reading:
- **Detector:** on these all-scaffold photos the bigger input boxes many more people (139 vs 106 of 200). End-to-end recall at a 1.5% false-alarm rate is close to zero for both arms, though. With 235 negatives, 1.5% means only 3 false alarms are allowed, and many not-on-scaffold people stand on decks, formwork or rebar right next to scaffolds, so they look the same to the judge (AP among boxed people 0.66–0.71). The bottleneck on hard scenes moves from the detector to the judge.
- **Helmet:** adding a zoomed head crop to the current crop is confirmed on new photos. Only 16 people without a helmet support it, so the CI is wide.

### 12.3 Overnight round (2026-11-22, validation only)
**F1_feet_crop → keep C1.** Adding a crop of the feet and support area to the judge gives +0.008 on the old val and +0.005 on the 120 new photos (AP among boxed people 0.713 vs 0.709). The likely limit is the training data (about 200 on-scaffold people, mostly not CS10k-style) and a frozen ImageNet encoder, not the crop. RUNLOG r00156–r00157.

**HL3_helmet_confirm → helmet change confirmed a second time.** The person boxes are CS10k's own human boxes. The 71 no-helmet people are rule-1 violators whose helmet state I checked on head crops; the 150 helmet-worn people are 6 violators wearing helmets plus 144 "white hard hat" boxes. RUNLOG r00158–r00159.

| Helmet head | Balanced acc. | No-helmet caught | Helmet correct |
|---|---|---|---|
| C0 crop | 0.749 | 70% | 80% |
| **C0 + head crop** | **0.827** | 73% | **92%** |
| Diff [95% CI] | **+0.077 [+0.028, +0.130]** | | |

Cost: one extra batched encoder pass, about 4.5 ms per image on the M5.
