---
title: Labeling v2 and controlled experiment setup
created: 2026-11-20
status: proposal v0.1 (nothing implemented yet)
tags: [genia, labeling, experiments, control, plan]
---
# Labeling v2 and the controlled experiment setup

**Asked by Luciano, 2026-11-20:** (1) first, do we need to control the optimization process? (2) define new labels from the conclusions of Codex and both paper reviews; (3) switch the experiments to the controlled setup, and check everything it touches.

**Built on:** Reminders (Fine-tuning · Quality over quantity, Evaluation · Human agreement, Dataset size, Labeling rows) · Labeling and dataset plan · [Labeling log](labeling_log.md) · [Ablation log](ablation_log.md) §3–10 · 03 Claude review of Codex analysis · 04 Claude - papers that solved the label problem · 05 Claude - relationship papers with the same abstractions · 07 Claude - review of note 06 and next experiment plan · 09 Claude - review of note 08 (Mac vs cloud).

**Frozen things stay frozen:** `labels/relations/v1/frozen_v1/` and `splits/v1/` are not touched. Everything below is **v2, added next to v1**.

---

## Part 1. Do we need to control the optimization process? **Yes, at three levels.**

**In simple words:** when we "optimize the labels", we try several label versions and keep the best. That choice is itself an optimization. If it isn't controlled, we can fool ourselves in three ways:

| Level | What gets optimized | How it fools us if uncontrolled | Control |
|---|---|---|---|
| **1. Training** (inside each run) | Model weights: epochs, learning rate, seed, early stopping, class weights | One arm gets a luckier seed or more training, and the difference looks like a label effect | **One fixed recipe for every arm**, written in a config before running. If tuning is needed, every arm gets **the same tuning budget on val**. Repeats: 5 draws/seeds per arm. Report the spread |
| **2. Decision** (the threshold) | The yes/no cutoff on the judge's score | Moving the threshold trades recall for false alarms (that is what happened with the classifier: R 0.43 → 0.53, FA 1.5% → 12%). Arms compared at different cutoffs can't be compared | **The threshold is chosen by one rule, on val only:** the cutoff where val false alarms = 1.5% (and 5%). Then report recall at that cutoff |
| **3. Selection** (across runs) | Which label policy / arm / input we keep | Trying many variants and picking the best **on the test set** overfits the test. With 191 test positives, a lucky +0.03 is noise. **This already happened in the §10 exploration** | **Pre-written decision rule**, selection on **val only**, **every run logged** (including failures and dropped ideas), **test opened once** for the single chosen winner |

**A fourth, easy-to-miss optimizer: the labeling loop itself.** If we write guideline rules or choose which photos to label *after looking at test errors*, the labels quietly adapt to the test. Controls: guideline changes are driven by **train/val** errors only; every labeled person records **how it was selected** (random / active / contrast pair), so a biased sample is visible.

[inference] Without these controls, a "better label policy" can't be told apart from "better luck, a different threshold, or a test set we tuned to".

---

## Part 2. Label schema v2 (relation_v2)

### What changes and why
| # | Change | Why (source) |
|---|---|---|
| 1 | Split `support` into **what supports the person** (`support_surface`) and derive on / not-on from it | v1 `none` lumped *ground* with *roof / lift / free ladder*. The at-height non-scaffold cases are the **hard negatives** we lack (07 Q2–3; TIN interactiveness, 05) |
| 2 | **`feet_on` + `feet_visible` required for every person**, not only RICH | Support is a **foot-contact** fact (Silberman 2012; HOT 2023; PaStaNet body-part states help most with few examples, 05) |
| 3 | **`proximity` required when the person is not on a scaffold and a scaffold is visible** | Names the hard-negative type (in_front_of is the key trap; Reminders · Hard negative) |
| 4 | **Ankle keypoints for every person**: pre-filled by a COCO pose model, then checked; hips dropped | Cheap support geometry (04: keypoint rule as in the hard-hat paper). Hand-placing 4 keypoints was the slowest RICH step |
| 5 | **Platform polygons required on every relation image**, including synthetic | The geometry feature needs them. v1 synthetic images had none (Labeling log §2) |
| 6 | **Harness rule tightened:** `not_worn` only if the shoulders and chest are visible with no straps, else `not_visible` | Harness κ was 0.39, mostly not_worn vs not_visible (Labeling log §9) |
| 7 | **`tied_off` parked:** only `anchor_visible: yes/no` is recorded, only for people with a harness worn. Not a training target | κ ≈ 0; 1 positive (Labeling log §9–10; 01 §7) |
| 8 | **Image-level `all_people_boxed: yes/no`** + optional **`ignore_region`** box for crowds or tiny people | Makes **person coverage** measurable and gives a true "ignore" (01 Control 1; 03 §4) |
| 9 | **Provenance per image:** `selected_by` (random / active / contrast / legacy_v1), `contrast_group` id, `annotator`, `pass` | Controls selection bias (Part 1, level 4); supports contrast pairs (Rel3D, 05) |
| 10 | **Four states are never merged:** yes / no / unclear / **unannotated**. An export never fills a missing field with "no" | 01 Q3; 03 §3.7 |

### Per-person fields (v2)
| Field | Values | Required | Notes |
|---|---|---|---|
| `support_surface` | `scaffold_platform` · `scaffold_frame_or_attached_ladder` · `ground` · `other_elevated` (roof, slab, formwork, lift, free-standing ladder, beam) · `unclear` | **Yes** | Replaces `support` |
| `feet_on` | `platform` · `frame_or_ladder` · `ground` · `other_surface` · `not_visible` | **Yes** | The contact fact. Usually agrees with `support_surface`, but can be `not_visible` while `support_surface` is decided from context |
| `feet_visible` | `both` · `one` · `none` | **Yes** | Lets us report results by visibility |
| `proximity` | `in_front_of` · `behind` · `under` · `next_to` · `far` · `no_scaffold_in_image` | Yes, if not on a scaffold | Hard-negative type |
| `platform_link` | relation person → platform polygon | Yes, if on a platform | As in v1 (`R` key) |
| `ankles` | left / right ankle keypoints, visibility 0/1/2 | Yes (pre-filled, checked) | From a COCO pose model, corrected only if clearly wrong |
| `helmet_state` | `worn` · `not_worn` · `held` · `not_visible` | Yes | Unchanged |
| `harness_state` | `worn` · `not_worn` · `not_visible` | Yes | Tightened rule (#6) |
| `anchor_visible` | `yes` · `no` | Only if harness worn | Replaces tied_off for now |
| `posture`, `height_band` | as v1 | Optional (RICH) | Kept for error analysis |

**Derived, never labeled directly** (as in `C_relation_v1.yaml`):
- `on_scaffold` = yes if `support_surface` ∈ {scaffold_platform, scaffold_frame_or_attached_ladder}; no if ∈ {ground, other_elevated}; unclear if unclear.
- **Backward map to v1** `support`: platform → `stands_on`, frame_or_ladder → `climbing`, ground / other_elevated → `none`, unclear → `unclear`. So v1 and v2 labels can be scored together.
- `hard_negative_type` = `support_surface` + `proximity` (e.g. ground + in_front_of; other_elevated + next_to).

### Per-image fields (v2)
`all_people_boxed` (required) · `ignore_region` (optional boxes) · `selected_by` · `contrast_group` · `site_or_sequence_id` (from source metadata) · v1 RICH fields (viewpoint, distance, lighting, occlusion) kept optional.

### Labeling process v2
1. **Pre-annotations:** person boxes (source + COCO pseudo boxes), ankles (COCO pose), platform boxes (our detector) as **editable suggestions**, never saved unchecked.
2. **AI labeling as in v1** (Claude first pass, Codex review), **plus a blind human check of ~50 people** (κ, and the fix for the "circular test" in 03 §3.4).
3. **Double label 15%** blind, as in v1. κ per field, with special attention to `support_surface` and `feet_on`.
4. **What to label** (from 07 Q2; no new photos): train split → archikhu, sparks, platform, climb (~360 imgs) + ~300 rf_harness_main imgs; val split → the same sources (for threshold selection); **CS10k-dev**: ~150 CS10k images not among the 159 test images, deduplicated. Plus **contrast pairs** where available (same scaffold, on vs in front).
5. **Re-label the existing v1 relation images to v2** only for the **new fields** (support_surface split, feet_on, feet_visible, proximity, ankles, all_people_boxed). The v1 support value is kept as a check: a v2 label that disagrees with v1 goes to review.

---

## Part 3. The controlled experiment setup

### 3.1 Data separation (fixed before any run)
| Set | Contents | Used for | Touched how often |
|---|---|---|---|
| **train_v2** | train split: v1 relation people + new v2 labels (scaffold sources + rf_harness_main) | Fitting the judge (detector fixed) | Freely |
| **val_v2** | val split: v1 relation people (36 decided) + new v2 val labels | Threshold (Part 1, level 2) and selecting the winner (level 3) | Every run |
| **cs10k_dev** | ~150 new CS10k images, disjoint from and near-duplicate-checked against the 159 test images | Hard-negative false-alarm rate for the threshold | Every run |
| **test_v1 (336 photos)** | 143 indist + 159 CS10k + 34 heldout: unchanged images and unchanged v1 labels | **Final report only**, each subset separately | **Once** per decision |

Frozen as `splits/v2/` with SHA256SUMS. `splits/v1/` is unchanged.

### 3.2 Fixed recipe (one config per study, in `GenIA-code/experiments/<date>_<name>/config.yaml`)
- Detector: fixed weights (step 1b base + V2 fine-tune) and fixed `conf`. Detections are cached once.
- Judge: input variant (C0–C3, 07 Q4), features cached; logistic regression with fixed L2 and solver; **outputs a probability**.
- Labels: label version (v1 / v2) and policy (drop unclear; harness from per-person field; Codex-reviewed).
- Synthetic: `real_only` or `synthetic_fixed` (two separate series, never mixed in one comparison).
- Repeats: 5 draws per arm (subset sampling grouped by image/sequence), fixed seed list `[0,1,2,3,4]`.
- Threshold rule: val + cs10k_dev false alarms = 1.5% (primary) and 5% (secondary).

### 3.3 The control metrics
| Control | Measures | Target or use | From |
|---|---|---|---|
| **C1 Person coverage** | audited people matched by a box ÷ all audited people (40–60 random photos, counted blind) | ≥ 95% as a process target (not a guarantee) | 01 Control 1; 03 §4 |
| **C2 Label agreement** | κ per field on the 15% double-labeled + 50-person human check | κ is the ceiling for the model; fields with κ < 0.6 aren't used as targets | Reminders · Human agreement |
| **C3 Class balance** | negatives per positive for on_scaffold in train (today 18:63 ≈ 0.29) + hard-negative share | Report per run; aim ≥ 1:1 | 03 §4 |
| **C4 Learning curve** | val recall at matched FA vs number of real training people (nested 25 / 50 / 100% of train_v2, 5 draws) | Flat **and** good → enough; flat **and** poor → representation problem; rising → keep labeling | Reminders · Quality over quantity; Dataset size |
| **C5 Error split** | on-scaffold people missed by the detector vs found but judged wrong | Decides whether to work on detection or judgement | 02 Q1; 07 Q5 |

### 3.4 Primary success measure and decision rule (written before running)
- **Primary:** on-scaffold **recall at val false-alarm rate 1.5%**, end-to-end (detector + judge), on val_v2 + cs10k_dev.
- **Also reported:** recall at 5%, conditional AP with true boxes, false alarms as **counts** ("9 of 620"), helmet and harness balanced accuracy, and bootstrap 95% CIs by image or sequence.
- **Rule:** an arm replaces the baseline only if its primary measure beats the baseline by **≥ 0.05** and the paired bootstrap interval excludes 0. Otherwise keep the baseline. Ties go to the simpler arm.
- **Test:** after the decision, run test_v1 once and report all three subsets. No changes after looking.

### 3.5 Studies in order (from 07, unchanged)
0. **Free checks:** the judge outputs probabilities, PR curves on val_v1, error split (C5), and inspect 30 `--neg 400` and 30 fypcv replay people.
1. **Context study** C0–C3 on current labels.
2. **Label v2** (Part 2) and measure C1–C3.
3. **Positive vs negative study:** R0, R+N⁻, R+N⁺, R+N±, R+2N (N = 100 as a budget), plus learning curve C4.
4. **Test once.**

### 3.6 Run log
Every run appends one line to `GenIA-code/experiments/RUNLOG.csv`: date, study, arm, draw, config hash, label version, split version, primary measure, CI, and whether it was kept. Failed and abandoned runs too. The vault gets a summary in [Ablation log](ablation_log.md) §11.

---

## Part 4. Files this touches (the "incumbencies")

**Nothing has been changed yet.** v1 files stay as they are. v2 versions are **added**.

| File | Today (v1) | v2 action |
|---|---|---|
| `GenIA-data/labeling_queue/relation_v1/GUIDELINES.md` | Rule, support values, edge cases, RICH fields | **New** `relation_v2/GUIDELINES.md` with the v2 fields, tightened harness rule, hard-negative examples, contrast pairs |
| `.../relation_v1/label_studio_config.xml` | `support` (4 values), proximity optional, RICH fields | **New** `relation_v2/label_studio_config.xml`: support_surface, required feet_on / feet_visible, proximity, ankles only, all_people_boxed, ignore_region, provenance |
| `.../relation_v1/queue.csv`, `label_studio_import.json` | 868 tasks | **New** v2 queue: v1 images (new fields only) + new train / val / cs10k_dev images, with pre-annotations |
| Label Studio project "GenIA relation v1" | 868 tasks | **New project** "GenIA relation v2". v1 untouched |
| `GenIA-code/configs/label_schemes/C_relation_v1.yaml`, `D_rich_v1.yaml` | support → on / next_to / unclear | **New** `C_relation_v2.yaml`, `D_rich_v2.yaml` with the derived on_scaffold and the backward map to v1 |
| `configs/label_schemes/rules.yaml` | uses `support`, `tied_off` | **New** `rules_v2.yaml`: on_scaffold from support_surface; `not_tied_off` rule disabled (anchor_visible only) |
| `02_labeling/claude_labeler/` (`write_ann.py`, `render.py`, `ppe_*`, `kappa.py`, `export_ann.py`) | v1 field names | Add v2 field names (κ for the new fields) |
| `02_labeling/codex_prompts/CONTEXT.md` | v1 rules | **New** `CONTEXT_v2.md` |
| `07_ablation/design_B.py`, `combined.py` | v1 `support` parse, argmax output, crop C0 | **New** `08_controlled/` folder: parse v1 + v2, probability output, cached detections, C0–C3 inputs, threshold rule, run log. 07 stays as the record of the ablation |
| `07_ablation/evaluate.py` | test scoring | Reused unchanged for the final test (v1 labels, 336 photos) |
| `GenIA-data/splits/v1/` | frozen | **New** `splits/v2/` (train_v2, val_v2, cs10k_dev, test_v1 list copied) |
| Vault: [Labeling log](labeling_log.md), [Ablation log](ablation_log.md), Reminders, Labeling and dataset plan, `DATA_CARD` | v1 | Add v2 sections (append, don't rewrite) |

## Part 5. Quantization (decided 2026-11-20)
**Decision:** quantization comes **after** the labeling optimization, as the last step on the winning model. It compresses the final model and changes neither the labels nor what the model learns. The judge is tiny, so quantization mainly concerns the YOLO detector (and a VLM later, if used). Three rules are fixed now:
1. **Same yardstick:** the quantized model is scored like every other arm, by on-scaffold recall at the matched false-alarm rate on val_v2 + cs10k_dev, with the same counts and CIs.
2. **Re-select the threshold after quantizing:** scores shift slightly after compression, so the 1.5% (and 5%) false-alarm cutoff is chosen again **on val** for the quantized model. The test stays untouched until the final single run.
3. **Calibration set:** INT8 calibration uses a few hundred representative **real** photos from train_v2 / val_v2, **never test_v1**. List them in `splits/v2/calibration.txt`.

## Interview line
"Once the labels became the thing I was optimizing, I controlled the optimization itself: one fixed training recipe for every arm, one threshold rule on validation, a decision rule written before the runs, and the test opened once. The labels moved from 'on or not' to the physical fact underneath, what the feet stand on, so the hard negatives become explicit and the model gets the contact cue the papers point to."
