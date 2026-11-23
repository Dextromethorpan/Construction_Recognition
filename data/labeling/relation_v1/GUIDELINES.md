# Labeling guidelines: person on scaffold (relation v1)

Labels describe **what is there**, not violations. Violations are computed later by `labels/schemes/rules.yaml`.
**When in doubt: `unclear`.** An honest `unclear` is worth more than a guess.

## 1. The rule

A person is **on the scaffold** when **both feet are on a scaffold platform above ground level**.
A ladder counts only if it is **attached to the scaffold** (then the person is `climbing`).

| support | when |
|---|---|
| `stands_on` | Both feet on a platform level (deck, plank, working platform) of the scaffold, above the ground. |
| `climbing` | On the scaffold frame or on a ladder attached to it, moving between levels. |
| `none` | Feet on the ground, a roof, a slab, a free-standing ladder, a lift, or anything that is not the scaffold. |
| `unclear` | Feet hidden and context does not decide it, image too small or blurry, or you hesitate. |

Scheme C is derived automatically: `stands_on`/`climbing` -> `on`, `none` -> `next_to`, `unclear` -> `unclear`.

Proximity (optional, multi): `in_front_of`, `behind`, `under`, `next_to`, relative to the nearest scaffold.

## 2. Per image, in this order (basic task, about 30 s)

1. Look at the pre-drawn boxes. They come from the original datasets and are **often incomplete**.
   Fix or delete wrong boxes. Add a `person` box for every visible person (head to feet, including occluded parts you can infer).
2. Draw each **walkable platform level** as a `scaffold_platform` polygon (the deck surface, not the whole frame).
   Keep `scaffold_structure` boxes around the whole scaffold if useful.
3. Click each person and set **support** (required).
4. For `stands_on` / `climbing`: link person -> platform with a relation (select person, select platform, `R`).
5. If the image has no person and no scaffold, tick `skip:no_person_no_scaffold` and submit.

## 3. RICH tasks (header says RICH, about 2 min)

Everything above, plus for **each person**: helmet state, harness state, tied off, height band, posture, visibility, feet_on,
and the **4 keypoints** (left/right ankle, left/right hip; place them where they are, even if occluded).
Per image: viewpoint, distance, lighting, occlusion.

- `helmet:held` = a hard hat in the hand or hanging from the belt. Caps, hoods and beanies are `not_worn`.
- `harness:worn` needs visible straps on the body; a lanyard alone is not enough.
- `tied:yes` only if you can see the lanyard attached to an anchor or lifeline.
- height_band: ground (0 m), low (< 2 m), mid (2 to 6 m), top (> 6 m or top lift), unknown.

## 4. Edge cases

| Situation | Label |
|---|---|
| Standing on the bottom plank, 20 cm above ground | `stands_on`, height `low` |
| Standing on the ground inside the scaffold frame | `none`, proximity `under` |
| In front of the scaffold, feet on ground, body overlaps the frame in 2D | `none`, `in_front_of` (the key hard negative) |
| On a free-standing A-frame ladder next to the scaffold | `none` |
| On a ladder tied to the scaffold | `climbing` |
| On a mobile scaffold platform | `stands_on` |
| On a roof, slab edge or formwork, no scaffold | `none` (relation hard negative) |
| Feet hidden behind a toe board or guardrail, body clearly above the deck | `stands_on`, visibility `feet_hidden` |
| Feet hidden and could be ground or deck | `unclear` |
| Mast climber / suspended cradle (gondola) | `stands_on` (it is a working platform), note it with occlusion/notes |
| Reflection, poster, or person far away (< 15 px tall) | do not box |

## 5. Double labeling

About 15% of images appear twice (`pass 2`, at the end of the list). Label them **again without looking at pass 1**.
Agreement (Cohen's kappa per predicate) is computed from these. Disagreements become `unclear` unless adjudicated.

## 6. Order of work

Labeling order = the task order (priority). 1: test images with people and scaffolds; 2: other test images;
3: val; 4: train (for the learning curve). Stop at any point: what is done is usable.

## 7. Example slots (fill with real screenshots as you go; aim for about 20)

| # | Case | Image (global_id) | Label |
|---|---|---|---|
| 1 | stands_on, clear | | |
| 2 | stands_on, feet hidden by toe board | | |
| 3 | climbing attached ladder | | |
| 4 | in_front_of, overlaps frame in 2D | | |
| 5 | under the scaffold | | |
| 6 | free-standing ladder | | |
| 7 | mobile scaffold | | |
| 8 | empty scaffold | | |
| 9 | roof work, no scaffold | | |
| 10 | unclear | | |
| 11-20 | your own hard cases | | |
