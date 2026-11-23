# Relation labels v2: frozen_v2 (frozen 2026-11-21)

Decision by Luciano, 2026-11-21 12:20 CET: **"Use now, flag the 203"**. Labels v2 are frozen now so the free checks and the
context study can start. The 203 tasks Codex has not reviewed yet are flagged `claude_only`. Codex finishes them on
2026-11-27; those fixes go into a new folder `frozen_v2.1/`, **never into this one**.

## Files
| File | What |
|---|---|
| `labels_v2.json` | Byte copy of `../export_20261121_111539.json` (Label Studio project 2 "GenIA relation v2", 1,367 tasks) |
| `people_v2.csv` | `parse_v2.py labels_v2.json --frozen-v1 frozen_v1/codex_reviewed.json` (identical to `../export_20261121_111539_people.csv`, checked with `cmp`) + a `review_status` column. One row per person box |
| `review_status.csv` | task_id, global_id, split_v2, pass, review_status |
| `kappa_pass2_PRE_review_official.csv` | **Official inter-annotator agreement**: Claude pass 1 vs Codex blind pass 2, before Codex reviewed pass 1 (`../kappa_pass2_before_review.csv`) |
| `kappa_pass2_POST_review_INFLATED.csv` | Same pairs after Codex reviewed pass 1 (`../kappa_pass2_final.csv`). **Inflated**: Codex wrote both sides. Do not quote as agreement |
| `human_check_sample.csv` | The 50-person sample for Luciano's own audit (seed 20261120) |
| `SHA256SUMS` | `shasum -a 256 -c SHA256SUMS` |
| `FROZEN` | Marker: nothing in this folder is edited |

## review_status
| Value | Tasks | Meaning |
|---|---|---|
| `codex_reviewed` | 986 | Claude pass 1, then reviewed by Codex (a `codex_reviews/<task>.json` exists: 831 fixed, 155 ok) |
| `claude_only` | 203 | Claude pass 1, **not reviewed yet** (202 train_v2, 1 val_v2; 199 new, 4 relabel_v1). 659 people |
| `pass2_codex_blind` | 178 | Codex blind second pass (κ copies). Not used for training or selection |

## Counts (pass 1, 1,189 images, 3,451 people)
| Set | Images | People | on | not_on | unclear |
|---|---|---|---|---|---|
| train_v2 | 818 | 2,289 | 357 | 1,652 | 280 |
| val_v2 | 221 | 681 | 100 | 499 | 82 |
| cs10k_dev | 150 | 481 | 53 | 384 | 44 |

support_surface: ground 2,166 · scaffold_platform 369 · other_elevated 369 · scaffold_frame_or_attached_ladder 141 · unclear 406.
`on_scaffold` is derived from support_surface (`C_relation_v2.yaml`). all_people_boxed = no on 146 people's images.
46 relabel people disagree with their v1 support (flag `disagrees_with_v1`).

## How it was made
Claude labeled all 1,189 pass-1 tasks with guideline v2.0 (written via `write_ann_v2.py`), Codex labeled the 178 pass-2
copies blind, then reviewed 986 of the pass-1 tasks; a harness recheck pass followed (`../harness_recheck_*.csv`).
Exported 2026-11-21 11:15 UTC.

## Known limits
- **203 claude_only tasks** (659 people, nearly all train_v2). Experiments can report with/without them; v2.1 replaces them.
- **κ < 0.6 before review** (official): proximity 0.35, feet_visible 0.55, anchor_visible 0.35 (n = 29). These are **not
  used as training targets** (decision_rule.md §6). on_scaffold κ 0.70, support_surface 0.68, helmet 0.89, harness 0.66.
- **CS10k small boxes:** cs10k_dev person boxes are much smaller than in train/val: median height 11.5% of the image
  (train 38.8%, val 40.0%) and 42% of them are under 10% high (train 4.8%). A detector and a crop-based judge trained on
  train_v2 see a different size regime there; expect lower detection recall and weaker crops on CS10k.
- All labels are AI-made (Claude + Codex). The human check sample is not done yet.
