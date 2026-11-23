---
title: Data Acquisition Agent log
tags: [genia, data, agent-log]
---
# Data Acquisition Agent log

## 2026-11-18 - Phase 1 (survey) done, STOPPED for go-ahead
- Surveyed 11 datasets (8 candidates; Roboflow harness and scaffold searches expanded to 5 sets). Nothing downloaded.
- Wrote `~/GenIA-data/manifest.yaml` and DATA_CARD.
- Prerequisites missing: `HF_TOKEN`, `ROBOFLOW_API_KEY`, Kaggle key. Only Python 3.9 is on the Mac; Phase 2 starts by creating the Python 3.11 venv in `~/GenIA-data/.venv` (Homebrew python@3.11).
- Skipped: MOCS (request form + display restrictions).
- Open questions for Luciano: go-ahead on the proposal; SODA (unclear license, manual Baidu download); SH17 (size likely > 5 GB).
- Survey was done by hand (web + APIs); no `01_survey.py` yet. The manifest is the record.

## 2026-11-18 - Phase 2 (download) done for 8 datasets
- Luciano approved the proposal. Downloaded into `~/GenIA-data/raw/` (7.1 GB total, budget 40 GB). All files verified against `raw/<id>/SHA256SUMS`; manifest updated.
- | id | version | images | size |
  |---|---|---|---|
  | shwd | VOC2028 zip | 7,581 (in zip) | 1.1 GB |
  | cs10k | HF rev ca3d9b88 (3 parquet) | 10,013 | 4.5 GB, sha256 matches HF LFS ids |
  | hardhat_wkrs | v2 raw | 7,035 | 0.25 GB |
  | rf_harness_main | v2 raw (no augmentation) | 9,875 | 0.74 GB |
  | rf_harness_mks | v2 (augmented only) | 7,880 files | 0.65 GB |
  | rf_harness_fypcv | v1 (augmented only) | 2,274 files | 0.21 GB |
  | rf_scaffold_platform | v1 raw | 179 | 0.01 GB |
  | rf_scaffold_climb | v7 raw | 872 | 0.05 GB |
- rf_harness_main was first fetched as v7 (3x augmented, 29k files); replaced by raw v2. The v7 copy is set aside in `~/GenIA-data/superseded/` (2 GB, safe to delete).
- rf_harness_mks and rf_harness_fypcv exist only as augmented exports: Phase 3 must group augmented copies by their original filename stem (Roboflow `<stem>_jpg.rf.<hash>`) so copies never span splits, and keep only one copy per stem.
- rf_scaffold_platform has only 179 images (page said ~1.8k) -> platform boxes will come mainly from hand labeling.
- CS10k was downloaded through Luciano's logged-in Chrome (no HF token needed). SHWD was run by Luciano after the auto-mode check blocked it.
- Still pending Luciano: SODA (unclear license, manual Baidu download), SH17 (size unknown, Kaggle key). MOCS skipped.

## 2026-11-18 - Phases 3-5 done, STOPPED at the split (not frozen yet)
- Phase 3: 8 sources -> `interim/<id>/annotations.coco.json` (38,262 images). Archived sources (SHWD zip, CS10k parquet) unpacked to `~/GenIA-data/extracted/` so raw/ stays untouched. 7,446 Roboflow augmented copies dropped (one per original stem), 20 boxes dropped (outside image / zero area). pyarrow installed with Luciano's OK.
- Phase 4: pHash, threshold 6, within and across sources. 8,554 near-duplicates removed; 2,142 clusters, 604 span two sources (rf_harness_main contains ~380 SHWD images and some hardhat/fypcv/mks ones). Greedy keepers so chained clusters (SHWD SCUT-HEAD classroom frames, 1,555 images) keep every non-duplicate image while staying one split group. Histogram in `reports/hamming_hist.csv`.
- Phase 5 (proposed): test_external = CS10k (9,394), test_heldout = rf_harness_mks (2,693), train 12,086 / val 2,687 / test_indist 2,715. 0 clusters span splits; 453 images excluded by the leak guard. Report: split_report_v1_proposed.
- GAP (stopping rule): only **123** images have people AND scaffolds labeled together (all from rf_scaffold_climb), below the 200 threshold. Harness is fine (1,678 images). CS10k has 1,180 images whose captions mention scaffolds and workers: good candidates for the hand-labeling queue (test side only). SODA would close the training gap.
- Splits NOT frozen: waiting on Luciano's decision about SODA first (adding it after freezing would force a v2).

## 2026-11-18 - SODA dropped, 5 scaffold sets added, split v1 FROZEN
- SODA skipped: only reachable through Baidu Netdisk with a Baidu account, which Luciano declined. Western-only sourcing was considered and dropped (no labeled Western worker+scaffold set exists on Roboflow; most existing sources are East Asian).
- Added (CC BY 4.0, raw versions): rf_scaffold_esco v2 (1,450), rf_scaffold_archikhu v1 (814 after stem dedup), rf_scaffold_sparks v2 (323), rf_scaffold_mobile v7 (573), rf_scaffold_hook v2 (100). raw/ now 7.9 GB.
- Dedup rerun: 41,522 images, 9,662 near-duplicates removed, 1,300 cross-source clusters (archikhu largely re-uploads the chungang sets).
- Gap closed: 1,990 images now label people and scaffolds together (train 1,396 / val 297 / test_indist 297); harness 1,678. No stopping rule triggered.
- Split v1 frozen (`splits/v1/SHA256SUMS`): train 13,597 / val 3,004 / test_indist 3,040 / test_heldout 2,692 (rf_harness_mks) / test_external 9,394 (cs10k). 0 clusters span splits. Report: [split_report_v1](split_report_v1.md).
- Next: Phase 6 (label schemes) and Phase 7 (relation labeling queue), after Luciano's OK at this checkpoint.

## 2026-11-18 - Phases 6-8 done (labeling prepared, nothing labeled)
- Phase 6: `labels/schemes/` A_objects_v1, B_objects_platform_v1, C_relation_v1, D_rich_v1 (YAML), VLM_schema_v1.json, rules.yaml. Missing vs absent uses the LVIS field `not_exhaustive_category_ids` per image. `scripts/export_scheme.py` built A and B for all four splits (`build/`, class counts in `reports/class_counts_*.csv`). Only 67 train images have real platform boxes -> platforms come from hand labels.
- Phase 7: `labeling_queue/relation_v1/`: queue.csv (754 images, 868 tasks incl. 114 blind second passes = 15%), all 7 buckets + 30 contrast pairs, priority 1 = all 296 test_indist/test_heldout images with person+scaffold. rich_D on all test images + 168 train. Label Studio config + import with existing boxes as editable predictions. GUIDELINES.md. CVAT not prepared (needs Docker).
- Phase 8: `splits/v1/learning_curve/` nested 10/50/200 per class, frozen (scaffold_platform has only 67).
- Waiting: Luciano's OK to `pip install label-studio` (separate venv `.venv-ls`).

## 2026-11-18 - Label Studio ready
- Label Studio 1.23.2 in `~/GenIA-data/.venv-ls`, started with `~/GenIA-code/02_labeling/label_studio.sh` on **http://localhost:8081** (8080 is taken by YouTrack in Docker). Database in `labeling_queue/label_studio_db/`.
- Project 1 "GenIA relation v1": config from `label_studio_config.xml`, 868 tasks imported (709 with pre-annotation boxes), local-file storages for raw/, extracted/ and labeling_queue/. Images load for all sources (checked Roboflow, CS10k).
- Code moved to the git repo `~/GenIA-code` (symlinks keep `~/GenIA-data/scripts` and `labels/schemes` working). See Project structure and environments.

## 2026-11-18/19 - Relation labeling done by AI (Claude + Codex)
- Luciano decided the labels are made by AI. Claude labeled all 754 first-pass tasks and re-checked PPE for 1,359 people. Codex labels the 114 blind second passes and reviews all 754, with one prompt per task and Claude supervising.
- ESCO `Scaffold-19..27` images are AI-generated (389 tasks). They are kept in test and reported separately.
- Full record: [Labeling log](labeling_log.md).
