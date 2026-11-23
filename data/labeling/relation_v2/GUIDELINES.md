# Labeling guidelines: person on scaffold (relation v2)

Version: **v2.0** (2026-11-20). Every change gets a new version line in §9 with its reason.
Plan: vault `Labeling/Labeling v2 and controlled experiment setup.md`. v1 rules (`../relation_v1/GUIDELINES.md`) are frozen.

Labels describe **what is there**, not violations. Violations are computed later by `configs/label_schemes/rules_v2.yaml`.
**When in doubt: `unclear` (or `not_visible`).** An honest `unclear` beats a guess.
**Never fill a missing field with "no".** A field you did not set stays *unannotated*; that is different from `no`.

## 1. The fact we label: what the feet stand on

v1 asked "on the scaffold or not?". v2 asks the physical fact underneath: **what surface supports the person**.
"On scaffold" is then **derived**, never labeled directly.

| `support_surface` | When | Derived `on_scaffold` | v1 `support` |
|---|---|---|---|
| `scaffold_platform` | Both feet on a deck, plank or working platform of a scaffold (incl. mobile scaffolds, mast climbers, suspended cradles), above ground | yes | `stands_on` |
| `scaffold_frame_or_attached_ladder` | On the scaffold frame, or on a ladder tied/attached to the scaffold | yes | `climbing` |
| `ground` | Feet on the ground (incl. standing on the ground inside or under the scaffold frame) | no | `none` |
| `other_elevated` | Above ground but **not** on a scaffold: roof, slab edge, formwork, beam, lift/MEWP basket, free-standing ladder, truck bed | no | `none` |
| `unclear` | Feet hidden and context does not decide, too small, too blurry, or you hesitate | unclear | `unclear` |

`other_elevated` is the **hard negative** v1 lacked: a person at height who is not on a scaffold.

## 2. Per-person fields

| Field | Values | Required | How to decide |
|---|---|---|---|
| `support_surface` | see §1 | **always** | Context may decide it even when the feet are hidden (e.g. body clearly above a toe board) |
| `feet_on` | `feet:platform` · `feet:frame_or_ladder` · `feet:ground` · `feet:other_surface` · `feet:not_visible` | **always** | Only what you **see** at the feet. `not_visible` if neither foot's contact point is visible |
| `feet_visible` | `fv:both` · `fv:one` · `fv:none` | **always** | Count feet whose contact with a surface you can see |
| `proximity` | `prox:in_front_of` · `prox:behind` · `prox:under` · `prox:next_to` · `prox:far` · `prox:no_scaffold_in_image` | **if support_surface is ground, other_elevated or unclear** | Position relative to the nearest scaffold. `far` = clearly away from it. `no_scaffold_in_image` = no scaffold visible |
| `helmet_state` | `helmet:worn` · `helmet:not_worn` · `helmet:held` · `helmet:not_visible` | **always** | Caps, hoods, beanies = `not_worn`. Hard hat in hand or on belt = `held` |
| `harness_state` | `harness:worn` · `harness:not_worn` · `harness:not_visible` | **always** | **Tightened (v2):** `not_worn` **only if shoulders and chest are visible and show no straps**. Back view with no straps visible, chest hidden, or person too small = `not_visible`. `worn` needs straps on the body; a lanyard alone is not enough |
| `anchor_visible` | `anchor:yes` · `anchor:no` | only if harness worn | Is the lanyard's anchor point (or lifeline) visible? This **replaces `tied_off`**, which is parked (κ ≈ 0, 1 positive) |
| platform link | relation person → platform polygon (`R` key) | if support_surface is `scaffold_platform` (and a polygon is drawn) | Relation value `on_platform`; `on_frame_or_ladder` for the frame/ladder case is optional |
| ankles | `left_ankle` / `right_ankle` keypoints; use `*_occluded` when you infer the position | yes (pre-filled, check) | Pre-filled by a COCO pose model. Correct only if **clearly** wrong (> half a foot off). Delete if the ankle is outside the image. Not placed = not labeled |
| `height_band`, `posture` | as v1 (`h:ground` … `h:unknown`; standing …) | optional | Kept for error analysis |

**Consistency checks** (the parser flags these for review, it does not fix them):
- `support_surface = scaffold_platform` with `feet_on = feet:ground` (or the reverse).
- `feet_visible = fv:none` with `feet_on` other than `feet:not_visible`.
- `support_surface` ∈ {ground, other_elevated, unclear} with no `proximity`.
- v2 `support_surface` that maps to a v1 `support` different from the frozen v1 label (relabel tasks only).

## 3. Per-image fields

| Field | Values | Required | Meaning |
|---|---|---|---|
| `all_people_boxed` | `boxed:yes` · `boxed:no` | **always** | `yes` = every visible person ≥ 15 px tall has a person box (or is inside an `ignore_region`). This makes **person coverage** measurable |
| `ignore_region` | rectangle(s) | optional | Crowds or tiny people you did not box one by one. Models are neither rewarded nor punished inside it |
| `scaffold_platform` polygons | polygon per walkable deck level | **on every image with a scaffold**, incl. synthetic | Deck surface only, not the whole frame |
| viewpoint, distance, lighting, occlusion | as v1 | optional | For error analysis |
| `image_flag` | `skip:no_person_no_scaffold` · `skip:bad_image` | if applies | |

Provenance (`selected_by`, `contrast_group`, `split_v2`, `task_kind`, `pass`) is **task data**, shown in the header. You do not label it.

## 4. Per image, in this order

1. Read the header: task kind (`relabel_v1` or `new`), pass, split.
2. Check the boxes (pre-filled from the source dataset and a COCO person model). Fix, delete, **add every missing person**. Use `ignore_region` for crowds/tiny people.
3. Draw / check `scaffold_platform` polygons (deck surface).
4. Click each person: `support_surface`, `feet_on`, `feet_visible`, then `proximity` if not on a scaffold, then helmet, harness (and `anchor_visible` if worn).
5. Link platform people to their polygon (`R`).
6. Check the two ankle points.
7. Set `all_people_boxed`. Submit.

**Relabel tasks (v1 images):** boxes, helmet and harness come from frozen v1 (Codex-reviewed). Only add the new fields (`support_surface`, `feet_on`, `feet_visible`, `proximity`, ankles, `all_people_boxed`) and fix harness where the tightened rule applies. The v1 support value is **not shown** (so it can't bias you). After labeling, `parse_v2.py --frozen-v1` compares your answer with v1; a disagreement is sent to review, never overwritten automatically.

## 5. Edge cases

| Situation | support_surface | feet_on | proximity |
|---|---|---|---|
| Bottom plank, 20 cm above ground | `scaffold_platform` | `feet:platform` | — |
| On the ground inside the scaffold frame | `ground` | `feet:ground` | `prox:under` |
| **In front of the scaffold, feet on ground, overlaps the frame in 2D** (the key hard negative) | `ground` | `feet:ground` | `prox:in_front_of` |
| Free-standing A-frame ladder next to the scaffold | `other_elevated` | `feet:other_surface` | `prox:next_to` |
| Ladder tied to the scaffold | `scaffold_frame_or_attached_ladder` | `feet:frame_or_ladder` | — |
| Mobile scaffold platform | `scaffold_platform` | `feet:platform` | — |
| Roof, slab edge or formwork, no scaffold in view | `other_elevated` | `feet:other_surface` | `prox:no_scaffold_in_image` |
| Roof edge next to a scaffold, not on it | `other_elevated` | `feet:other_surface` | `prox:next_to` |
| Lift / MEWP basket | `other_elevated` | `feet:other_surface` (or `not_visible`) | as seen |
| Feet behind a toe board, body clearly above the deck | `scaffold_platform` | `feet:not_visible` | — |
| Feet hidden, could be ground or deck | `unclear` | `feet:not_visible` | as seen |
| Mast climber / suspended cradle | `scaffold_platform` | as seen | — |
| Behind the scaffold (scaffold between camera and person), on the ground | `ground` | as seen | `prox:behind` |
| Reflection, poster, person < 15 px tall | do not box (use `ignore_region` if many) | | |

## 6. Contrast pairs

When two images (or two people in one image) show the **same scaffold** with one person on it and one in front of / next to it,
they share a `contrast_group` id (set in the queue). Label each person independently. These pairs teach the boundary.

## 7. Double labeling and human check

- About 15% of tasks appear twice (`pass 2`, at the end of the list). Label them **again without looking at pass 1**.
- A blind **human check of ~50 people** (Luciano) is done on a random sample. κ per field is computed from both.
- A field with κ < 0.6 is not used as a training target until this guideline is fixed.

## 8. Order of work

Task order = priority: (1) relabel v1 val + train images (new fields), (2) new val images, (3) cs10k_dev, (4) new train images
(scaffold sources first, then rf_harness_main), (5) pass 2. Within a source the order is random (seeded), so stopping early leaves a random sample, not a biased one.

## 9. Version history

| Version | Date | Change | Reason (train/val evidence only) |
|---|---|---|---|
| v2.0 | 2026-11-20 | First v2: support_surface split, feet_on/feet_visible required, proximity required off-scaffold, ankles only, tightened harness, anchor_visible replaces tied_off, all_people_boxed, ignore_region, provenance | Labeling log §9–10; review notes 01–09 |
