# Study 0: free checks (2026-11-21), selection data only (val_v2 + cs10k_dev, real only). No test data.

Judge = C0 (judge.py, draw 0) on the fixed detector `runs/ablation_v2b/V2_harness_field/weights/best.pt` (imgsz 512, conf 0.25).

## C3 class balance (train_v2, real_only, frozen_v2)
on 245 · not_on 1,638 · dropped 250 unclear/unannotated → **6.7 negatives per positive** (v1 was 0.29). Hard negatives
(other_elevated) 134 = 8.2% of negatives; ground people near a scaffold 272 = 16.6%. 532 of 1,883 usable people are claude_only.

## C5 error split (C0, draw 0; 118 on-scaffold people in val_v2 + cs10k_dev)
FA 1.5%: found 55 · **detector miss 35** · judge miss 28 (detector 56% of misses). FA 5%: detector 61% of misses.
Detector misses by set: val_v2 12/65 on-people never get a box (82% reachable); cs10k_dev 23/53 (only 57% reachable),
matching the small-box finding (cs10k_dev median person height 11.5% of the image vs 40% in val).

## PR curve, C0 judge (pr_C0_d0.png)
| Input boxes | Set | pos / neg | AP |
|---|---|---|---|
| labeled (conditional) | val_v2 + cs10k_dev | 118 / 878 | 0.661 |
| labeled | val_v2 | 65 / 494 | 0.902 |
| labeled | cs10k_dev | 53 / 384 | 0.303 |
| detector (end-to-end) | val_v2 + cs10k_dev | 118 / 878 | 0.583 |
| detector | val_v2 | 65 / 494 | 0.754 |
| detector | cs10k_dev | 53 / 384 | 0.343 |
CS10k is the hard part for both stages: the judge itself is weak there (AP 0.30 with true boxes) and the detector misses 43%.

## Inspection by eye (sheet_neg400.jpg, sheet_fypcv_replay.jpg, zoom_fypcv.jpg; seed 20261121)
- **30 of the old `--neg 400` people (hardhat_wkrs / shwd):** 0 clearly at height (2 unsure: #5 on a slab edge, #28 by a
  drill rig). But ~9 of 30 are not construction at all (classrooms, offices, a poster, mannequins in a mine): easy negatives
  that teach little. Keep `extra_negatives: none`.
- **30 rf_harness_fypcv replay people (V2 fine-tune train, 160 people in 96 images):** about **7 of 30 are at height**:
  #0/#7/#21 (one video: elevated gangway along a facade, scaffold-like), #1 (roof edge), #3 (steel truss), #13 (rooftop
  deck), #22 (inside a tower/crane lattice). These carried no on-scaffold label in the detector fine-tune, so the
  detector's own `person_on_scaffold` class saw them as implicit negatives. The judge does not use that class; do not
  reuse fypcv people as "not on scaffold" negatives without labeling them.

## C1 coverage proxy (true C1 needs Luciano's blind audit)
all_people_boxed = no on 25 train images, 0 val, 0 cs10k_dev. Detections with no labeled person (IoU < 0.3, conf ≥ 0.5):
train 17, val 5, cs10k_dev 9 → few unlabeled people left.

## Found: the fixed detector never predicts scaffold_platform
0 class-4 boxes at conf 0.25 on all 1,041 images (max conf 0.22 on the test predictions from 07). This made study 1's C2 a no-op.
