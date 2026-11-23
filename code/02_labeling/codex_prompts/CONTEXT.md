# Shared context for every Codex labeling prompt

You are labeling construction-site images for the GenIA project (person on scaffold + PPE).
Label Studio 1.23.2 runs locally at http://localhost:8081, project id 1 "GenIA relation v1".
You do NOT use the web UI. You write labels as JSON spec files and push them with a script.

## Hard rules
1. Touch only the task named in your prompt.
2. Never move, rename or delete anything in `~/GenIA-data` (paths are frozen; splits have checksums).
3. Never edit Label Studio's database by hand. Only use `write_ann.py` as shown below.
4. Always run with `ANNOTATOR=codex` so your work is recorded separately from Claude's.
5. Second-pass tasks are BLIND: do not open `claude_specs/`, any `export_*.json`, or `claude_labeled_tasks.csv`
   for that image. Label from the image only.
6. When in doubt, use `unclear`. An honest `unclear` beats a guess.
7. Full guidelines: `~/GenIA-data/labeling_queue/relation_v1/GUIDELINES.md` (read once, it is short).

## Tools (all in `~/GenIA-code/02_labeling/claude_labeler/`)
- Render the image with a % grid and the original dataset boxes:
  `~/GenIA-data/.venv/bin/python render.py TASK_ID` (add `--crop x0,y0,x1,y1` in percent to zoom).
  Output path is printed; open the JPG to look at it.
- Draw your spec over the image to check it: `~/GenIA-data/.venv/bin/python check.py SPEC.json`
- Write to Label Studio:
  `ANNOTATOR=codex SPEC_FILES=/abs/path/NNNN.json zsh ls_env.sh $PWD/write_ann.py`
  Expect a line `ok TASK_ID ...`. Add `"replace": true` in the spec to overwrite an existing annotation.

## Spec format (coordinates are PERCENT of image width/height, origin top-left)
```json
{"task": 123,
 "boxes": [
  {"id":"p1","label":"person","x":40,"y":10,"w":12,"h":35,
   "support":"stands_on","on":"pl1","proximity":["next_to"],
   "helmet_state":"helmet:worn","harness_state":"harness:worn","tied_off":"tied:yes",
   "height_band":"h:mid","posture":"standing","visibility":"vis:full","feet_on":"feet:platform",
   "kp":{"left_hip":[44,28],"right_hip":[48,28],"left_ankle":[44,44],"right_ankle":[48,44]}},
  {"id":"s1","label":"scaffold_structure","x":30,"y":0,"w":60,"h":90},
  {"id":"h1","label":"helmet","x":45,"y":10,"w":4,"h":4},
  {"id":"hn1","label":"harness","x":42,"y":16,"w":8,"h":14}
 ],
 "platforms":[{"id":"pl1","points":[[30,45],[90,45],[90,48],[30,48]]}],
 "viewpoint":"view:ground","distance":"dist:mid","lighting":"light:day","occlusion":"occ:partial"}
```
Box labels: person, helmet, head, harness, scaffold_structure, ladder, fence, rebar, formwork.
Use `"on":"plX"` only when support is stands_on or climbing and a platform polygon exists.
Optional image flag: `"image_flag":"skip:bad_image"` or `"skip:no_person_no_scaffold"`.

## Field values (per person)
- support (required): stands_on | climbing | none | unclear
  - stands_on: both feet on a scaffold deck/plank above ground. climbing: on the scaffold frame or a ladder attached to it.
  - none: feet on ground, roof, slab, free-standing ladder, lift. In front of the scaffold but feet on the ground = none + in_front_of.
- proximity (optional, list): in_front_of | behind | under | next_to
- helmet_state: helmet:worn | helmet:not_worn | helmet:held | helmet:not_visible (caps/hoods/beanies = not_worn)
- harness_state: harness:worn (straps visible on body) | harness:not_worn | harness:not_visible
- tied_off: tied:yes (lanyard visibly attached to anchor/lifeline) | tied:no | tied:not_visible
- height_band: h:ground | h:low (<2 m) | h:mid (2-6 m) | h:top (>6 m) | h:unknown
- posture: standing | climbing_posture | kneeling | sitting | leaning
- visibility: vis:full | vis:partial | vis:feet_hidden
- feet_on: feet:platform | feet:ground | feet:ladder | feet:not_visible
- keypoints (only if you can see the joints; omit rather than guess): left_ankle, right_ankle, left_hip, right_hip
Per image: viewpoint (view:ground|elevated|aerial), distance (dist:near|mid|far),
lighting (light:day|low|night|backlit), occlusion (occ:none|partial|heavy).

## The three things that matter most on every image
1. Helmet: is each person wearing one? (and draw a `helmet` box if missing)
2. Safety harness: straps visible? lanyard tied off? (draw a `harness` box if missing)
3. Scaffold relation: stands_on / climbing / none / unclear, plus a platform polygon under people who stand on it.

## Note on image sources
Images whose id contains `Scaffold-19` to `Scaffold-27` are AI-generated (listed in
`~/GenIA-data/labels/relations/v1/synthetic_images.csv`). Label them the same way; they are reported separately.

## Status of Claude's labels (2026-11-18, end of day)
- All 754 first-pass tasks are labeled by Claude. Every person (1,359) had helmet, harness and tied-off checked on a zoomed crop.
- Keypoints exist only on the first ~125 RICH test tasks (hand-placed). CS10k tasks have none, on purpose.
- Synthetic ESCO images (`synthetic_images.csv`) got quick labels: support + image fields, no platform polygons, no keypoints.
- `tied:yes` was set only when the lanyard's anchor point is clearly visible; otherwise `tied:not_visible`. Check this on review.
- Tasks 306, 311, 488, 502, 525, 531 had misaligned person boxes: redraw them.
