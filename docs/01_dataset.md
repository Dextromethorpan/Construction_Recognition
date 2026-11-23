# Dataset

**Images are not included in this repository.** Run `code/01_data/` to download, convert, deduplicate and split them. Then check the result against the manifests and the SHA256 checksums in `data/splits/`. What *is* included are our own relation labels (`data/labels/`), the labeling guidelines and the label schemes.

## Sources (13 used)
| id | Dataset | Used for | Licence |
|---|---|---|---|
| hardhat_wkrs | Hard Hat Workers (Roboflow Universe) | helmet, head, person | Public Domain (per Roboflow) |
| shwd | Safety Helmet Wearing Dataset (GitHub, njvisionpower) | helmet, bare head | MIT on the repo; image copyright unclear |
| rf_harness_main | harness-s4xxh/harness-usugr | harness, person, helmet | CC BY 4.0 |
| rf_harness_fypcv | fypcv/safety-harness-iess4 | harness | CC BY 4.0 |
| rf_harness_mks | mks-temp/scaffold-harness | **held-out test source** | CC BY 4.0 |
| rf_scaffold_platform | bowens-workspace-3mivc/scaffold-yolov8-detection | working platforms | CC BY 4.0 |
| rf_scaffold_climb | chungang-likkc/scaffold-golkd | ladders, climbing | CC BY 4.0 |
| rf_scaffold_esco | esco-7veqt/scaffold-7bi3u | scaffolds and workers (part is AI-generated, see below) | CC BY 4.0 |
| rf_scaffold_archikhu | archikhu/scaffold-0726-ddew4 | scaffolds and workers | CC BY 4.0 |
| rf_scaffold_sparks | sparks-f4jed/scaffold-a4l14 | scaffolds and workers | CC BY 4.0 |
| rf_scaffold_mobile | chungang-likkc/mobile-scaffold | mobile scaffolds | CC BY 4.0 |
| rf_scaffold_hook | objectdetectionproject-vp2dx/worker-scaffold-hook | hooks | CC BY 4.0 |
| cs10k | ConstructionSite-10k (Hugging Face, LouisChen15/ConstructionSite; images from MOCS) | **external test set** | CC BY-NC 4.0, gated, display restrictions. Not redistributed and not shown in figures |

Three datasets were skipped:
- **SODA:** available only through Baidu Netdisk, and no licence is stated.
- **MOCS:** needs a request form and has display restrictions.
- **SH17:** has no harness class, and it is manufacturing rather than construction.

## Processing
1. **Convert** every source to one COCO-style format, keeping the original class names. 7,710 Roboflow augmentation copies were dropped (one per original).
2. **Deduplicate** with perceptual hashing (pHash, Hamming distance ≤ 6), within and across sources: 9,662 near-duplicates removed, 1,300 clusters span two sources. A whole duplicate cluster always stays in one split.
3. **Split by source and site, not by photo** (seed 20261118). One whole source, rf_harness_mks, is held out as `test_heldout`, and CS10k is `test_external`. No duplicate cluster spans two splits, and 456 images were excluded by the leak guard. Split v1 is frozen with checksums.
4. **Label schemes** are small mapping files over one frozen image pool (`data/label_schemes/`). Every comparison therefore uses the same images and the same test split.

## Split v1 (detector data, frozen 2026-11-18)
| Split | Images | Notes |
|---|---|---|
| train | 13,597 | 11 sources |
| val | 3,004 | same sources |
| test_indist | 3,040 | same sources, different sites |
| test_heldout | 2,692 | rf_harness_mks only |
| test_external | 9,394 | CS10k only |

Person boxes in the sources are sparse: many people have no box. A COCO-pretrained YOLO11m (confidence ≥ 0.5) added 34,459 person boxes in train and 8,219 in val. See `docs/02_labels.md`.

## Relation labels (our own)
| Version | Images | People | Use |
|---|---|---|---|
| v1 (frozen 2026-11-19) | 754 (+114 blind second passes) | 1,670 | Labeling-policy ablation; **the test set labels** (336 real test photos) |
| v2 (frozen 2026-11-21) | 1,189 (+178 blind second passes) | 3,451 | Training and validation for the controlled studies |
| v2x (2026-11-22) | 120 extra photos | 504 | Validation-only confirmation after the test |

**Split v2** (frozen 2026-11-20), used only for the relation judge:
| Split | Images | People |
|---|---|---|
| train_v2 | 818 | 245 real on-scaffold, 1,638 real not-on |
| val_v2 | 221 | — |
| cs10k_dev | 150 | not in a duplicate cluster with any test image |

The test set stayed **test_v1**: 336 real photos (143 in-distribution, 159 CS10k, 34 held-out).

**Per-person fields** (v2): support surface, from which "on scaffold" is derived; feet on and feet visible; proximity; helmet; harness; anchor visible; ankle keypoints. The full definitions are in `data/labeling/relation_v2/GUIDELINES.md`.

**Agreement** (Cohen's kappa, blind second pass, measured before the review):
| Field | v1 | v2 |
|---|---|---|
| Helmet | 0.80 | 0.89 |
| On scaffold | 0.69 | 0.70 |
| Harness | 0.39 | 0.66 |
| Proximity | — | 0.35 (not used as a target) |

## Known issues and biases
- **Synthetic images:** ESCO `Scaffold-19` to `Scaffold-27` are AI-generated (389 labeled tasks, listed in `data/labels/v1/synthetic_images.csv`). They are reported separately, and all headline numbers use real photos only.
- **The held-out source** is mostly augmented video frames with noise and watermarks. Treat it as a robustness check.
- **rf_scaffold_mobile** is mostly frames from one rooftop video.
- **Origin bias:** almost all sources show East Asian sites or are web-scraped. No labeled Western worker-on-scaffold set was found.
- **SHWD** negatives are largely classroom heads, not construction.
- **The labels are AI-made** (Claude first pass, Codex review). 203 v2 images were labeled by Claude only and are flagged. No human audit has been done yet.

## Attribution
Roboflow Universe sets: CC BY 4.0, by the workspaces named above. Hard Hat Workers: Northeastern University China, via Roboflow. SHWD: njvisionpower (GitHub). ConstructionSite-10k: Chen et al., Hugging Face LouisChen15/ConstructionSite, with images from MOCS (Xuehui et al., Automation in Construction 2021).
