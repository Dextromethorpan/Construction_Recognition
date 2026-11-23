# Shared context for every Codex labeling prompt (relation v2)

v1 context (`CONTEXT.md`) is frozen and still describes project 1. This file is for **project "GenIA relation v2"** only.
You are labeling construction-site images for GenIA (person on scaffold + PPE). Label Studio runs at http://localhost:8081.
You do NOT use the web UI. You write labels as JSON spec files and push them with a script.

## Hard rules
1. Touch only the task named in your prompt.
2. Never move, rename or delete anything in `~/GenIA-data`. Never touch `labels/relations/v1/` or `splits/v1/` (frozen).
3. Never edit Label Studio's database by hand. Only use `relation_v2/write_ann_v2.py` as shown below.
4. Always run with `ANNOTATOR=codex`.
5. Pass-2 tasks are BLIND: do not open `labels/relations/v2/claude_specs/`, any export, or `claude_labeled_tasks.csv` for that image.
6. When in doubt: `unclear` / `not_visible`. **Never set a field to "no" because you did not check it**: leave it out instead.
7. Full rules: `~/GenIA-data/labeling_queue/relation_v2/GUIDELINES.md` (v2.0). Read it once.
8. On relabel tasks the boxes, helmet and harness come pre-filled from frozen v1. The v1 `support` is deliberately not shown, and you must not look it up in `labels/relations/v1/`. Give **your own** v2 answer; disagreements are found and reviewed afterwards.

## Tools
- Render (project 2 task ids): `~/GenIA-data/.venv/bin/python ~/GenIA-code/02_labeling/relation_v2/codex_v2/render.py TASK_ID` (`--crop x0,y0,x1,y1` in % to zoom).
  Check a spec: `... codex_v2/render.py --spec /abs/path/NNNN.json` -> overlay with every person's fields printed. Images land in `relation_v2/codex_v2/_render/`.
  (Do NOT use `claude_labeler/render.py`: it reads the v1 dump and shows the wrong image for v2 task ids.)
- Write: `ANNOTATOR=codex SPEC_FILES=/abs/path/NNNN.json zsh ~/GenIA-code/02_labeling/claude_labeler/ls_env.sh ~/GenIA-code/02_labeling/relation_v2/write_ann_v2.py`
  Expect `ok TASK_ID ...`. Add `"replace": true` to overwrite.
- Specs go in `~/GenIA-data/labels/relations/v2/codex_specs/NNNN.json`.

## Spec format (percent of image width/height, origin top-left)
```json
{"task": 123, "guideline_version": "v2.0",
 "boxes": [
  {"id":"p1","label":"person","x":40,"y":10,"w":12,"h":35,
   "support_surface":"scaffold_platform","on":"pl1",
   "feet_on":"feet:platform","feet_visible":"fv:both",
   "helmet_state":"helmet:worn","harness_state":"harness:worn","anchor_visible":"anchor:no",
   "height_band":"h:mid","posture":"standing",
   "kp":{"left_ankle":[44,44],"right_ankle_occluded":[48,44]}},
  {"id":"p2","label":"person","x":5,"y":40,"w":10,"h":50,
   "support_surface":"ground","proximity":"prox:in_front_of",
   "feet_on":"feet:ground","feet_visible":"fv:both",
   "helmet_state":"helmet:worn","harness_state":"harness:not_visible"},
  {"id":"s1","label":"scaffold_structure","x":30,"y":0,"w":60,"h":90}
 ],
 "platforms":[{"id":"pl1","points":[[30,45],[90,45],[90,48],[30,48]]}],
 "ignore":[{"x":80,"y":60,"w":15,"h":10}],
 "all_people_boxed":"boxed:yes"}
```

## Field values
Per person (required unless noted):
- `support_surface`: scaffold_platform | scaffold_frame_or_attached_ladder | ground | other_elevated | unclear
  (other_elevated = roof, slab, formwork, beam, lift basket, free-standing ladder: at height but NOT a scaffold)
- `feet_on`: feet:platform | feet:frame_or_ladder | feet:ground | feet:other_surface | feet:not_visible (only what you SEE)
- `feet_visible`: fv:both | fv:one | fv:none
- `proximity` (required if support_surface is ground / other_elevated / unclear): prox:in_front_of | prox:behind | prox:under | prox:next_to | prox:far | prox:no_scaffold_in_image
- `helmet_state`: helmet:worn | helmet:not_worn | helmet:held | helmet:not_visible
- `harness_state`: harness:worn | harness:not_worn | harness:not_visible. **`not_worn` only if shoulders AND chest are visible with no straps.**
- `anchor_visible` (only if harness worn): anchor:yes | anchor:no. There is no tied_off field in v2.
- optional: height_band (h:ground|low|mid|top|unknown), posture.
- `on`: platform polygon id, when support_surface is scaffold_platform.
- `kp`: ankles only. Use `*_occluded` when you infer the position. Omit rather than guess.
Per image: `all_people_boxed` (boxed:yes | boxed:no) required; `ignore` boxes for crowds/tiny people; optional viewpoint, distance, lighting, occlusion, image_flag (as v1).

## The key hard negatives
- Feet on the ground in front of the scaffold, overlapping it in 2D → `ground` + `prox:in_front_of`.
- At height on a roof / slab / lift / free ladder → `other_elevated` (+ proximity). This is NOT on scaffold.
