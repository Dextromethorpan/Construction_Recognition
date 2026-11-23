---
title: Labeling log
created: 2026-11-19
tags: [genia, labeling, log]
status: in progress
---
# Labeling log (relation v1)

This is the full record of how the relation labels were produced: decisions, numbers, files and open issues. Use it when recapitulating the project.
Related: Labeling and dataset plan · Reminders · [agent_log](agent_log.md) · DATA_CARD · [split_report_v1](split_report_v1.md)

## 1. Timeline and decisions

| When (local) | What | Decided by |
|---|---|---|
| 2026-11-18 evening | Label Studio running at localhost:8081, project "GenIA relation v1", 868 tasks (754 images + 114 blind second passes) | setup |
| 2026-11-18 ~19:35 | **Labels are made by AI, not by hand.** Claude labels the first pass | Luciano |
| 2026-11-18 ~20:15 | 46, then about 339, ESCO images (`Scaffold-19` to `Scaffold-27`) found to be **AI-generated** | Claude found it |
| 2026-11-18 ~20:20 | **Keep the synthetic images in test**, and report them separately | Luciano |
| 2026-11-18 ~20:05 | CS10k keypoints removed and marked missing, not estimated, because fake joints are worse than none | coordinator |
| 2026-11-18 ~22:40 | Luciano will not label the 114 second passes. **Codex does them**, plus a review of every Claude label, with one prompt per task | Luciano |
| 2026-11-18 ~23:05 | Claude re-checks helmet, harness and support for every person on zoomed crops | Luciano asked |
| 2026-11-18 ~23:20 | Claude supervises Codex through an automatic loop | Luciano asked |
| 2026-11-19 ~09:40 | The loop had stopped overnight after 19 tasks (the session ended). It was restarted detached, with the Mac kept awake | Claude |
| 2026-11-19 ~10:45 | **Codex does the full review of all 754 images (no sampling).** Document everything for a later recap | Luciano |

## 2. How the labels were made

**Claude, first pass (754 images)**
- For each task Claude rendered the image with a % grid (`claude_labeler/render.py`), wrote a JSON spec (`labels/relations/v1/claude_specs/NNNN.json`) and pushed it into Label Studio with `write_ann.py`. It writes through Label Studio's Django layer, because the legacy API token is disabled.
- **Real-photo test images:** detailed labels, with person boxes, platform polygons, support, PPE fields and hand-placed keypoints (first ~125 RICH tasks).
- **CS10k (test_external):** boxes drawn by eye, with no keypoints.
- **Synthetic ESCO images:** quick labels (support and image fields), with no polygons and no keypoints.
- **PPE re-check:** all 1,359 people were viewed on zoomed crops (`ppe_sheet.py`, `ppe_apply.py`). About 900 values were corrected and about 35 boxes that weren't people were deleted.
- Every task Claude wrote is listed in `claude_labeled_tasks.csv`.

**Codex, second opinion**
- The prompts are in `~/GenIA-code/02_labeling/codex_prompts/`. `pass2/` holds 114 blind labels, used to measure agreement. `review/` holds 754 checks of Claude's labels. The shared rules are in `CONTEXT.md`.
- `supervise.sh` runs the Codex CLI that ships inside the ChatGPT app (`codex exec`), one task at a time with the image attached, at about 1 minute per task.
- Codex output goes to `codex_specs/`, `codex_reviews/` and `codex_labeled_tasks.csv`, and each run is logged in `codex_prompts/runs/`. Claude's specs are never edited, but a Codex fix replaces Claude's annotation in Label Studio.

**Claude, supervising**
- Spot-checks Codex by drawing its specs on the image (`check.py`) and confirms every run saved.
- `kappa.py` matches people between the two passes (IoU ≥ 0.3) and computes Cohen's kappa for support, helmet, harness and tied-off.

## 3. Numbers (as of 2026-11-18 22:05 export)

- 754 first-pass tasks labeled, with 1,359 people.
- **Support:** stands_on 411, climbing 229, none 617, unclear 102.
- **Helmet:** worn 982, not worn 202, not visible 175.
- **Harness:** worn 235, not worn 970, not visible 154.
- **Tied off:** yes 1, no 1,023, not visible 335. Tied-off is only marked "yes" when the anchor point is visible.
- 12 images flagged `skip:bad_image`, mostly corrupted harness_mks video frames.
- Test set: 536 images (test_indist 334, test_external 159, test_heldout 43). 171 of them are synthetic, all in test_indist.

## 4. Data-quality findings to remember

- **Synthetic images:** ESCO `Scaffold-19` to `Scaffold-27` are AI-generated, 389 tasks in total, listed in `synthetic_images.csv`. Almost everyone in them is on the scaffold, so they add almost no negative examples. Report metrics on all images and on real photos only.
- **test_heldout (rf_harness_mks):** mostly augmented video frames with noise, watermarks and "FOR PREVIEW" overlays. Treat it as a robustness check, not a headline number.
- **rf_scaffold_mobile:** mostly frames from one rooftop video, so it inflates counts but adds little variety.
- **Misaligned person boxes** on tasks 306, 311, 488, 502, 525, 531. They're flagged in the Codex review prompts.

## 5. Token usage (2026-11-19 10:40)

- **Codex:** 67 tasks used about 1.7 M tokens (~25k per task). The full remaining queue (~800 tasks) is about 20 M more, on the ChatGPT plan.
- **Claude:** 3% of the Pro weekly limit. The labeling conversation held about 770k tokens in context, mostly images.
- **Possible savings, not applied, because Luciano chose the full review:** review a sample (~150 images), skip synthetic images, lower Codex reasoning effort, start a fresh thread for each phase.

## 6. Files

| What | Where |
|---|---|
| Label Studio (run) | `~/GenIA-code/02_labeling/label_studio.sh` → http://localhost:8081 |
| Guidelines | `~/GenIA-data/labeling_queue/relation_v1/GUIDELINES.md` |
| Claude specs and log | `~/GenIA-data/labels/relations/v1/claude_specs/`, `claude_labeled_tasks.csv` |
| Codex specs, reviews and log | `.../codex_specs/`, `codex_reviews/`, `codex_labeled_tasks.csv` |
| Snapshots | `~/GenIA-data/labels/relations/v1/export_*.json` (never overwritten) |
| Synthetic list | `~/GenIA-data/labels/relations/v1/synthetic_images.csv` |
| Claude tools | `~/GenIA-code/02_labeling/claude_labeler/` |
| Codex prompts, loop and logs | `~/GenIA-code/02_labeling/codex_prompts/` (`supervise.sh`, `runs/`, `STOP` to pause) |

## 7. Still open

- Codex blind second pass (114 tasks), then the agreement score (kappa), to be added here.
- Codex review of all 754 Claude labels, with a summary of how many it fixed and what kind of errors.
- Human audit of a small sample (optional, recommended for the interview).
- None of the labeling code is committed to git yet (`02_labeling/claude_labeler/`, `codex_prompts/`).

## 8. Say it in the interview

"The labels were produced by one model and cross-checked by an independent second model. Their agreement is my label-quality estimate, synthetic and corrupted sources are reported separately, and a human audit of a sample is the next step before production."

## 9. Agreement result: Claude vs Codex, blind second pass (2026-11-19 12:23)

Computed with `kappa.py` on export `export_20261119_112341.json`. The sample is 114 double-labeled images. People were matched by box overlap (IoU ≥ 0.3), giving 186 matched people.

| Field | Cohen's kappa | Raw agreement | Reading |
|---|---|---|---|
| Support (on scaffold) | **0.69** | 79% | Substantial. Most disagreements are Codex saying `unclear` where Claude gave a definite answer, not contradictions. Only 4 cases of climbing vs stands_on |
| Helmet | **0.80** | 92% | Strong |
| Harness | 0.39 | 63% | Weak. 47 cases are Claude `not_worn` vs Codex `not_visible`, a judgement on hidden straps. 14 are Codex seeing a harness Claude marked not worn, which are real Claude misses |
| Tied off | ~0.0 | 20% | **A convention mismatch, not a perception problem.** Claude defaulted to `tied:no` and Codex to `tied:not_visible`. The field is unusable as-is |

- **Person coverage:** Claude boxed 195 people and Codex boxed 254. Codex found about 59 more people, mostly small or background ones, so Claude's first pass under-counts people.
- **Takeaways:** support and helmet labels are reliable enough to use. Harness needs a clearer rule for when straps can't be seen. Tied-off should be redefined, for example as "lanyard anchor visible: yes or no", or dropped.

## 10. Codex review of all 754 first-pass tasks (done 2026-11-19 15:57)

Ran with 4, then 8, parallel workers. There was a ~1 h stall from an internet drop, and a watchdog plus auto-retry was added. Final export: `export_20261119_145823.json`.

- **Verdicts:** 716 fixed, 38 accepted as-is. Codex changed something in 95% of tasks, though often only small box or polygon refits.
- **People:** about 311 person boxes added in net (mostly small or background people Claude missed). The first pass now has **1,670 people** (was 1,359).
- **Support:** 180 changed. Mostly definite → `unclear` (stands_on→unclear 55, climbing→unclear 27, none→unclear 17), plus climbing↔stands_on swaps (42).
- **Helmet:** 93 changed. Mostly `not_visible` → `worn` (46). In 11 cases Codex saw no helmet where Claude said worn.
- **Harness:** 403 changed. Mostly Claude `not_worn` → Codex `not_visible` (327), plus **39 harnesses Claude missed**.
- **Tied off:** 807 changed from `no` to `not_visible`, adopting Codex's convention.

**Label distribution after the review (first pass, 1,670 people)**
- **Support:** stands_on 395, climbing 194, none 841, unclear 240.
- **Helmet:** worn 1,188, not worn 269, not visible 210, held 3.
- **Harness:** worn 270, not worn 626, not visible 774.
- **Tied off:** yes 1, no 212, not visible 1,457. This is too sparse to be useful as a target.

**Notes for the recap**
- Label Studio now shows **Codex's version** wherever it fixed something. Claude's originals are kept in `claude_specs/` and the older `export_*.json` snapshots.
- The rise in `unclear` and `not_visible` makes the labels more honest but reduces usable positive and negative examples. Decide whether to exclude `unclear` from training and evaluation.
- Harness "not visible" now dominates, so for a harness detector rely on the `harness` **boxes** rather than the per-person state.

## 11. Labeling v2 files built (2026-11-20)

Plan: [Labeling v2 and controlled experiment setup](labeling_v2_and_controlled_experiment_setup.md). Every file and its purpose: Labeling v2 - file map. The loop: [Labeling v2 - optimization loop](labeling_v2_optimization_loop.md).
**Nothing was labeled or trained.** v1 files were checksummed before and after: unchanged.

**Built**
- Rules and interface: `GenIA-data/labeling_queue/relation_v2/GUIDELINES.md` (v2.0, with a version table) and `label_studio_config.xml` (support_surface, required feet_on / feet_visible / helmet / harness, proximity, anchor_visible, ankles only, ignore_region, all_people_boxed).
- Schemes: `C_relation_v2.yaml` (on_scaffold derived; backward map to v1), `D_rich_v2.yaml`, `rules_v2.yaml` (tied-off rule off; new "no harness at height, not scaffold" rule). Codex rules: `codex_prompts/CONTEXT_v2.md`.
- Tools in `GenIA-code/02_labeling/relation_v2/`: `v2_fields.py` (one definition of all fields), `parse_v2.py` (v1 or v2 export → one table per person; never turns missing into "no"; flags inconsistencies), `kappa_v2.py`, `write_ann_v2.py` (writes only to the v2 project), `build_splits_v2.py`, `build_queue.py`, `prefill_pose.py` (built, not run).
- Control machinery in `GenIA-code/08_controlled/`: `decision_rule.md` (written before any run), `config_template.yaml`, `threshold.py`, `runlog.py`, plus `experiments/RUNLOG.csv` (header only).

**splits/v2 (FROZEN 2026-11-20, approved by Luciano)**
| Set | Images | Content |
|---|---|---|
| train_v2 | 818 | 168 relabel v1 (113 of them synthetic) + 350 new scaffold-source images (archikhu 261, sparks 52, platform 20, climb 17; ~917 prefilled people) + 300 rf_harness_main (~997 people) |
| val_v2 | 221 | 50 relabel v1 (35 synthetic) + 71 new scaffold-source images + 100 rf_harness_main (~355 people) |
| cs10k_dev | 150 | Stratified like the 159 CS10k test images (62 at height without scaffold, 57 scaffold with workers, 28 look-alike, 3 cap/hood). None in a duplicate cluster with test; 5 more removed for pHash distance ≤ 10 |
| test_v1 | 336 | 143 indist + 159 CS10k + 34 heldout, exactly as `evaluate.py` selects them |

**Queue v2:** 1,189 images, 1,367 tasks (178 blind pass-2 copies, 15% per group). Order: relabel val (50) → relabel train (168) → new val (171) → cs10k_dev (150) → new train scaffold (350) → new train harness (300) → pass 2.
Pre-fills: relabel tasks get the frozen v1 boxes, polygons, helmet and harness but **not** the v1 support (so it can't anchor the new answer). New tasks get source + COCO pseudo boxes. Pass-2 copies get boxes only, so κ isn't inflated by a shared suggestion. cs10k_dev has no boxes until `prefill_pose.py` runs.

**Checks run:** split checksums OK; no image in two v2 sets; every v2 image stays in its v1 split; `parse_v2` + `kappa_v2` on the frozen v1 export give on-scaffold κ 0.78 over 244 matched people (pass 1 vs pass 2, IoU ≥ 0.5); `runlog.py` refuses a test run without a decision and a second test run; `threshold.py` refuses test rows.

**Found on the way:** the v1 `rules.yaml` is not valid YAML (an unquoted colon in `rule_1_violation`). It is frozen, so it was left as is and fixed in `rules_v2.yaml`.

**Needs Luciano's OK before it happens:** freeze splits/v2, run `prefill_pose.py` (downloads yolo11n-pose.pt), create the Label Studio project "GenIA relation v2" and import the queue, start the AI labeling.

### 11b. Pose prefill and Label Studio project v2 (2026-11-20, 15:30)
Approved by Luciano ("continue"). Run on the Mac (MPS) because the Linux workspace has no PyTorch.
- **Pose prefill:** `prefill_pose.py` with yolo11n-pose (saved in `GenIA-data/runs/`) added **166 person boxes and 3,588 ankle points over 1,189 images**, in about 20 s. Output: `labeling_queue/relation_v2/label_studio_import_prefilled.json` (the original import file is kept). Two small fixes went into `prefill_pose.py`: (1) when two pose detections matched one person box, only the best one adds ankles (this avoids duplicate result ids); (2) a clean model_version name.
- **Label Studio project "GenIA relation v2" = project id 2:** 1,367 tasks (train_v2 818 + 122 pass 2, val_v2 221 + 34, cs10k_dev 150 + 22). 1,260 tasks carry suggestions; 107 have none to give. The config passed Label Studio's validation. Settings and the three local-files storages are copied from project 1.
- **Note:** Label Studio shows only the predictions whose model_version matches the project's, so every suggestion in project 2 is stored as `relation_v2_prefill`. The original names (B_persons / frozen_v1 / pose) are kept in the JSON file.
- **Checks:** one image per set opened (HTTP 200) with its suggestions. Project 1 is unchanged: 868 tasks, same model_version.
- **Not started:** the AI labeling. It needs Luciano's OK.

## 12. Labels v2 frozen (2026-11-21)
Luciano chose "Use now, flag the 203". `GenIA-data/labels/relations/v2/frozen_v2/` holds a copy of `export_20261121_111539.json` (labels_v2.json), people_v2.csv with a review_status column, review_status.csv (986 codex_reviewed · 203 claude_only · 178 pass2_codex_blind), κ tables (pre-review = official; post-review = inflated), human_check_sample.csv, README, SHA256SUMS and FROZEN (read-only). Pass 1: 3,451 people (train 357 on / 1,652 not_on / 280 unclear; val 100 / 499 / 82; cs10k_dev 53 / 384 / 44). Codex's fixes to the 203 on Nov 27 go to `frozen_v2.1/`. Studies: [Ablation log](ablation_log.md) §11.
