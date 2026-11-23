# GenIA demo — 9 photos

Pipeline = the final tested one (experiments/20261123_final_test/test_final.py), draw d=0: detector D0 coreml_int8 (yolo11n, 512 px, conf 0.25, CPU) → per person: C1 context crop → on-scaffold head (on if prob ≥ cutoff 0.367 = val false-alarm budget 1.5%); C0 person crop → helmet / harness heads (worn if prob ≥ 0.5). Encoder: frozen yolo11n-cls. Heads: GenIA-code/09_demo/heads_d0.npz.

Box colour: green = on scaffold, grey = not. Label: helmet · harness · scaffold, ✓/✗ + probability.

All 9 photos are real photos from the val_v2 split (no CS10k, no synthetic images, not from the test set). Truth = frozen_v2 pass-1 labels. Values like `not_visible` / `unclear` are not scored in the study.

Regenerate: `cd ~/GenIA-code && .venv-train/bin/python 09_demo/make_presentation.py`.

Licences: all images come from Roboflow Universe datasets under CC BY 4.0 (attribution per photo below).

## 01_helmet_success.png — Helmet: success

- Source id: `rf_scaffold_archikhu/hard_hat_workers253` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/train/hard_hat_workers253_png.rf.9caba9337aec35328ba44cc248b361d9.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: two hard hats, both found
- Run: 2 person box(es), detector 5 ms, total 20 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| pre0 | worn / not_worn / ground (not_on) | Y 0.96 / N 0.00 / N 0.00 |
| pre1 | worn / not_worn / ground (not_on) | Y 0.96 / N 0.00 / N 0.00 |

## 02_helmet_correct_no.png — Helmet: correct 'no helmet'

- Source id: `rf_scaffold_archikhu/49` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/train/49_jpg.rf.3397215d1292b4a05fd8398300b09d37.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: no one wears a helmet: 4 of 5 correct, 1 false helmet. Also a scaffold contrast: 2 on the deck flagged, 3 on the floor beside it not
- Run: 5 person box(es), detector 5 ms, total 26 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| b1 | not_worn / not_visible / ground (not_on) | N 0.00 / N 0.05 / N 0.13 |
| b2 | not_worn / not_visible / scaffold_platform (on) | N 0.00 / Y 0.56 / Y 1.00 |
| b3 | not_worn / not_visible / ground (not_on) | Y 0.85 / N 0.00 / N 0.00 |
| b4 | not_worn / not_worn / ground (not_on) | N 0.06 / N 0.00 / N 0.00 |
| b5 | not_worn / not_visible / scaffold_platform (on) | N 0.00 / Y 0.74 / Y 1.00 |

## 03_helmet_failure.png — Helmet: failure

- Source id: `rf_scaffold_esco/R-scaffold-new-7` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_esco/test/R-scaffold-new-7_jpg.rf.ac639f126bd04596301cef56fc659bb9.jpg`
- Licence: CC BY 4.0 — scaffold (ESCO), Roboflow Universe esco-7veqt/scaffold-7bi3u (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: white helmet + face mask on a worker carrying a deck board: missed
- Run: 2 person box(es), detector 5 ms, total 17 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| b1 | worn / not_visible / unclear (unclear) | Y 1.00 / N 0.00 / N 0.00 |
| b2 | worn / not_visible / scaffold_platform (on) | N 0.01 / N 0.21 / N 0.02 |

## 04_harness_success.png — Harness: worn at height

- Source id: `rf_scaffold_esco/R-scaffold-114` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_esco/test/R-scaffold-114_jpg.rf.e60c16a5b8d1c75647130191ab5a3aac.jpg`
- Licence: CC BY 4.0 — scaffold (ESCO), Roboflow Universe esco-7veqt/scaffold-7bi3u (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: full-body harness on a scaffold deck
- Run: 1 person box(es), detector 5 ms, total 22 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| b1 | worn / worn / scaffold_platform (on) | Y 0.99 / Y 0.81 / Y 1.00 |

## 05_harness_correct_no.png — Harness: correct 'no harness'

- Source id: `rf_scaffold_archikhu/4` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/test/4_jpg.rf.c1cf99c576d91c952d77920320020041.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: four workers, none with a harness
- Run: 4 person box(es), detector 6 ms, total 27 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| pre0 | worn / not_worn / other_elevated (not_on) | Y 1.00 / N 0.00 / N 0.00 |
| pre1 | worn / not_worn / other_elevated (not_on) | Y 1.00 / N 0.05 / N 0.00 |
| pre2 | worn / not_visible / other_elevated (not_on) | Y 1.00 / N 0.00 / N 0.08 |
| pre3 | worn / not_visible / other_elevated (not_on) | Y 1.00 / N 0.00 / N 0.00 |

## 06_harness_failure.png — Harness: failure

- Source id: `rf_scaffold_archikhu/119` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/train/119_jpg.rf.a5eca229fe3a366d286d94127be33fcc.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: harness worn by a small, distant worker at the top of a tower: missed
- Run: 1 person box(es), detector 29 ms, total 48 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| pre1 | worn / worn / scaffold_platform (on) | Y 0.96 / N 0.00 / Y 0.99 |

## 07_scaffold_success.png — On scaffold: success

- Source id: `rf_scaffold_esco/R-scaffold-new-30` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_esco/test/R-scaffold-new-30_jpg.rf.b67a4a90b28ce8ddec05f1283d850305.jpg`
- Licence: CC BY 4.0 — scaffold (ESCO), Roboflow Universe esco-7veqt/scaffold-7bi3u (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: three workers on decks, all flagged
- Run: 3 person box(es), detector 5 ms, total 23 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| b1 | not_visible / not_visible / scaffold_platform (on) | Y 0.63 / Y 1.00 / Y 1.00 |
| b2 | not_visible / not_visible / scaffold_platform (on) | Y 0.86 / Y 1.00 / Y 1.00 |
| b3 | not_worn / not_visible / scaffold_platform (on) | N 0.08 / Y 0.94 / Y 1.00 |

## 08_scaffold_correct_ground.png — On scaffold: correct 'ground beside'

- Source id: `rf_scaffold_archikhu/601` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/test/601_jpg.rf.dbf992b87eb520a6c4fada3b199c2f3f.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: man on the floor next to the mobile tower: not flagged (grey); man on the deck: flagged (green). Left man: support labelled unclear
- Run: 3 person box(es), detector 5 ms, total 22 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| b1 | not_visible / not_visible / scaffold_platform (on) | N 0.34 / N 0.26 / Y 1.00 |
| b2 | not_visible / not_visible / ground (not_on) | Y 0.53 / N 0.00 / N 0.00 |
| b3 | worn / not_visible / unclear (unclear) | Y 0.83 / N 0.00 / N 0.08 |

## 09_scaffold_failure.png — On scaffold: failure

- Source id: `rf_scaffold_archikhu/452` (split `val_v2`, real photo)
- File: `~/GenIA-data/raw/rf_scaffold_archikhu/train/452_jpg.rf.c257ee8c4fd7c0e6aa21b94dc0c18779.jpg`
- Licence: CC BY 4.0 — Scaffold 0726 (Archikhu), Roboflow Universe archikhu/scaffold-0726-ddew4 (https://creativecommons.org/licenses/by/4.0/)
- Why chosen: worker climbing the tower frame gets no person box, so no alert
- Run: 0 person box(es), detector 6 ms, total 6 ms

| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |
|---|---|---|
| pre1 | not_worn / not_visible / scaffold_frame_or_attached_ladder (on) | no box (detector miss) |
