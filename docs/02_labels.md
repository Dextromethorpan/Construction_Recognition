# How the labels were made and changed


## The principle we kept all along
**We label facts, not violations.** A label says what is in the photo ("on a platform, harness not worn"). A rule file (`rules.yaml`) computes the violation afterwards. We took this from the construction scene-graph papers (Liu 2023, Yi 2026), and it means a new safety rule never needs relabeling.

Four answers are never merged: **yes, no, unclear, and unannotated**. A field nobody filled in is never turned into "no".

## Stage 0. Labels we inherited (the public datasets)
- The 13 source datasets had boxes for helmets, harnesses, scaffolds and sometimes people. None of them labels "person on scaffold" as a relationship.
- **Problem found later:** many people had no box at all, so the detector learned that people are background (16% of people found).
- **Change:** a COCO person detector filled in 34,459 missing person boxes (people found: 16% → 55%). This was a machine fix, not a hand fix. The papers on sparse annotation (Niitani 2019, Co-mining 2021) describe the same mechanism.

## Stage 1. Relation labels v1 (18 to 19 Nov)
**How they were made**
- **18 Nov, your decision:** the labels are made by AI, not by hand. Claude labeled all 754 images in Label Studio: person boxes, platform polygons, support (stands on / climbing / none / unclear), helmet, harness and tied-off.
- **Your request:** Claude re-checked helmet and harness for all 1,359 people on zoomed crops (about 900 values corrected).
- **Your decision:** you would not label the 114 blind second passes yourself, so **Codex** did them, without seeing Claude's answer. Then you asked Codex to **review all 754** images, not a sample, with Claude supervising.

**What the agreement showed (kappa, Claude vs Codex blind)**
| Field | Kappa | Meaning |
|---|---|---|
| Helmet | 0.80 | Reliable |
| Support (on scaffold) | 0.69 | Usable |
| Harness | **0.39** | Weak: one said "not worn", the other "not visible" for hidden straps |
| Tied off | about 0 | Unusable: two different conventions |

**What the Codex review changed:** 716 of 754 images were corrected. It added about 311 missed people (1,359 → 1,670), moved many definite answers to "unclear", changed 327 harness "not worn" to "not visible", and found 39 harnesses Claude had missed.

## Stage 2. Choosing a labeling policy by experiment (19 Nov)
We did not argue about the policy; we trained one model per policy and compared them. The result, which you accepted:
- **Drop "unclear" people** from training (keeping them as a class or calling them "not on" did not help, and "not on" gave the worst false alarms).
- **Harness from the per-person field**, not from harness boxes (0.70 → 0.75).
- **Codex-reviewed labels**, with one honest caveat: they were scored against Codex's own labels, so this proves the review *filled gaps*, not that it is *more correct* (Claude's review, note 03).

## Stage 3. Why we decided to rewrite the rules (20 Nov)
Three inputs came together, and you asked for a v2 plan from them:
1. **Our numbers:** harness kappa 0.39, tied-off about 0, and only 88 real training people for the scaffold judgement.
2. **Codex's paper review (notes 01, 02)** and **Claude's critique of it (note 03)**. The critique found the cheapest fix: most labeled real people sat in the test split, so put labels where training can use them.
3. **Claude's two paper searches:** papers that solved parts of the label problem (note 04) and papers with the same abstraction outside construction (note 05). Their main message: "on" is a **support and contact** fact, so label **where the feet are**, and label hard negatives on purpose.

## Stage 4. Labels v2: what changed in the rules (GUIDELINES v2.0, 20 Nov)
| Change | Why |
|---|---|
| **Ask what supports the person** (`support_surface`: scaffold platform / scaffold frame or attached ladder / ground / other elevated surface / unclear). "On scaffold" is **derived** from it, never labeled directly | v1 "none" mixed people on the ground with people on roofs, slabs and lifts. Those at-height non-scaffold people are the hard negatives we lacked |
| **`feet_on` and `feet_visible` required for every person** | Support is a foot-contact fact (Silberman 2012, HOT 2023, PaStaNet) |
| **`proximity` required when someone is not on a scaffold but one is visible** (in front of / behind / under / next to / far) | Names the hard-negative trap, especially "in front of" |
| **Ankle keypoints** pre-filled by a pose model and then checked | Cheap geometry, as in Safe-Construct and the hard-hat keypoint paper |
| **Harness rule tightened:** "not worn" only if shoulders and chest are visible with no straps, otherwise "not visible" | Fixes the 0.39 disagreement |
| **Tied-off parked:** only "anchor visible yes/no" is kept, and it is not a training target | Kappa about 0 and 1 positive example |
| **`all_people_boxed` per image** and an optional ignore region for crowds | Makes person coverage measurable |
| **Provenance per image** (`selected_by`: random / active / contrast / legacy) | Controls selection bias |
| **Rule: a field with kappa below 0.6 is not a training target** | Agreement is the ceiling for the model |

**How changes are agreed and recorded:**
- Every guideline change gets a version line in GUIDELINES §9 with its reason, and the reason must come from **train or validation errors only**, never from test.
- A relabel is applied to train and validation alike.
- When v1 images were relabeled, their old "support" answer was **hidden**, so it could not anchor the new answer. A v2 answer that disagreed with v1 went to review.
- Blind second-pass copies got person boxes only, so agreement was not inflated by a shared suggestion.
- You approved each step that changed data: freezing the v2 splits, the pose prefill, creating the Label Studio v2 project and starting the AI labeling.

**Result:** 1,189 images and 3,451 people. Real on-scaffold people in training went from 63 to 245, and not-on people from 18 to 1,638. Agreement: helmet 0.89, on scaffold 0.70, harness **0.66** (was 0.39), proximity 0.35 (too low, so not used as a target). These are the honest numbers, measured before the Codex review. Labels were frozen as `frozen_v2` on 21 Nov, with 203 images flagged as Claude-only because Codex ran out of credits.

## Stage 5. Labels after the test (21 and 22 Nov, validation only)
- **120 new photos** were labeled to confirm two candidates (detector and helmet). Rules were written before any image was chosen. One amendment after 8 images: choose by caption ("on/climbing a scaffold") because positives were rare. **Claude only**, no second labeler. This is disclosed.
- **221 CS10k people with the dataset's own human boxes** were used for the second helmet check.
- **Machine-made person boxes for 2,000 CS10k photos** were tried and rejected: the teacher model misses the same people, so it marked them as background again.


