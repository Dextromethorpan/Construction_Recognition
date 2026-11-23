"""Build Codex prompts for project 2 (GenIA relation v2), modelled on codex_prompts/pass2 and review (v1, unchanged).
  codex_prompts/v2_pass2/task_NNNN.md  : blind labeling of every pass-2 task (178)
  codex_prompts/v2_review/task_NNNN.md : review of every Claude pass-1 task (claude_specs/NNNN.json)"""
import json, os
from urllib.parse import unquote
H = os.path.expanduser('~/GenIA-data'); C = os.path.expanduser('~/GenIA-code/02_labeling')
T = json.load(open(f'{H}/labeling_queue/relation_v2/claude_tasks_dump_v2.json'))
V = f'{H}/labels/relations/v2'; PY = '~/GenIA-data/.venv/bin/python'
RENDER = '~/GenIA-code/02_labeling/relation_v2/codex_v2/render.py'
WRITE = 'zsh ~/GenIA-code/02_labeling/claude_labeler/ls_env.sh ~/GenIA-code/02_labeling/relation_v2/write_ann_v2.py'
# people whose crops looked like no person during the harness re-pass (harness_recheck_box_suspect_idx.txt)
SUSPECT = {1353: ['a0', 'a1'], 1358: ['a0', 'a1'], 1372: ['a0', 'a1', 'a2']}
FIELDS = """   - `support_surface`, `feet_on`, `feet_visible`, `proximity` (if ground / other_elevated / unclear),
     `helmet_state`, `harness_state` (**not_worn only if shoulders AND chest are visible with no straps**; back views,
     side views, bent-over or tiny people -> `harness:not_visible`), `anchor_visible` (only if harness worn),
     `on` + a `platforms` polygon when support is scaffold_platform, ankles in `kp` only where you can see them."""


def head(t):
    d = t['data']; img = os.path.join(H, unquote(d['image'].split('?d=')[1]))
    return (f"Task {t['id']} | image `{d['global_id']}` | {d['task_kind']} | split {d['split_v2']} | priority {d['priority']} | pass {d['pass']}\n"
            f"Image file: `{img}`\n\nFirst read `~/GenIA-code/02_labeling/codex_prompts/CONTEXT_v2.md` (rules, schema, commands) and, once,\n"
            f"`~/GenIA-data/labeling_queue/relation_v2/GUIDELINES.md` (v2.0).\n")


def pass2(t):
    n = t['id']; spec = f'{V}/codex_specs/{n:04d}.json'
    rel = ("\nThis is a relabel task: boxes, helmet and harness are pre-filled from frozen v1. Check them against the image and give your own\n"
           "v2 answer for every field. Do not look anything up in `labels/relations/v1/`.\n") if t['data']['task_kind'] == 'relabel_v1' else ''
    return head(t) + f"""
# Your job: label this second-pass task BLIND (project 2, "GenIA relation v2")
The same image was labeled once by Claude in a pass-1 task. Your labels are compared with Claude's to measure agreement (kappa),
so do NOT look at Claude's labels: no `labels/relations/v2/claude_specs/`, no `export_*`, no `claude_labeled_tasks.csv`.
{rel}
Steps
1. `{PY} {RENDER} {n}` and look at the image and the person crops (pre-filled boxes are drawn with their ids).
   Zoom with `--crop x0,y0,x1,y1` (percent) on every person, especially head, shoulders/chest and feet.
2. Write `{spec}` (`"task": {n}`, `"guideline_version": "v2.0"`):
   - a `person` box for every visible person (reuse/adjust the pre-filled boxes; delete boxes on non-persons; add missing people;
     use `ignore` for crowds or people under ~15 px tall),
   - for each person:
{FIELDS}
   - per image: `all_people_boxed` (boxed:yes / boxed:no), and `image_flag` only if the image is unusable.
   When in doubt: `unclear` / `not_visible`. Never set "not_worn"/"no" because you did not check.
3. `{PY} {RENDER} --spec {spec}`, open the overlay and fix anything misplaced.
4. `ANNOTATOR=codex SPEC_FILES={spec} {WRITE}` -> expect `ok {n} ...`.
5. Report: number of people, support per person, helmet/harness per person, anything you were unsure about.
"""


def review(t):
    n = t['id']; cs = f'{V}/claude_specs/{n:04d}.json'; spec = f'{V}/codex_specs/{n:04d}.json'; rv = f'{V}/codex_reviews/{n:04d}.json'
    sus = (f"\nNote: during Claude's harness re-pass, the crops of person(s) {', '.join(SUSPECT[n])} looked like they contain no person.\n"
           "Check those boxes first: delete them if there is no person, or move them onto the person.\n") if n in SUSPECT else ''
    rel = ("\nThis is a relabel task: boxes, helmet and harness came pre-filled from frozen v1. Do not look anything up in `labels/relations/v1/`.\n"
           ) if t['data']['task_kind'] == 'relabel_v1' else ''
    return head(t) + f"""
# Your job: review Claude's v2 labels on this task and fix what is wrong
Claude's spec: `{cs}` (this is what is in Label Studio now).
{sus}{rel}
Steps
1. `{PY} {RENDER} --spec {cs}` and open the overlay (every person's fields are printed below the path).
   Also `{PY} {RENDER} {n}` and zoom with `--crop x0,y0,x1,y1` on every person (head, shoulders/chest, feet).
2. Check, person by person:
   - every visible person has a box (add missing ones, delete boxes on non-persons, fix boxes that miss the person),
   - helmet_state matches the image (a cap, hood or welding mask is not a helmet),
   - harness_state follows the strict rule: not_worn ONLY if shoulders AND chest are visible with no straps; otherwise
     not_visible unless straps are actually seen (worn). Also check anchor_visible when worn,
   - support_surface / feet_on / feet_visible / proximity are right (feet on the ground in front of a scaffold = ground +
     prox:in_front_of; roof, slab, ladder, lift, beam = other_elevated), and the platform polygon + `on` link when on a scaffold deck,
   - ankle keypoints sit on the ankles (delete wrong ones rather than guess), and `all_people_boxed` is right.
3. If everything is right, write `{rv}` as
   `{{"task": {n}, "verdict": "ok", "notes": ""}}` and stop.
4. If something is wrong: copy the spec to `{spec}`, fix it, set `"replace": true`, run
   `{PY} {RENDER} --spec {spec}` to check it, then `ANNOTATOR=codex SPEC_FILES={spec} {WRITE}` (expect `ok {n} ...`).
   Then write `{rv}` with `"verdict": "fixed"` and a list of the changes in `"notes"` (person id, field, old -> new, why).
5. Never edit the file in `claude_specs/`; your fixed copy goes in `codex_specs/`. Touch only task {n}.
"""


if __name__ == '__main__':
    for kind, fn, sel in (('v2_pass2', pass2, lambda t: t['data']['pass'] == 2),
                          ('v2_review', review, lambda t: t['data']['pass'] == 1 and os.path.exists(f"{V}/claude_specs/{t['id']:04d}.json"))):
        d = f'{C}/codex_prompts/{kind}'; os.makedirs(d, exist_ok=True); k = 0
        for t in T:
            if sel(t): open(f"{d}/task_{t['id']:04d}.md", 'w').write(fn(t)); k += 1
        print(kind, k)
    os.makedirs(f'{V}/codex_specs', exist_ok=True); os.makedirs(f'{V}/codex_reviews', exist_ok=True)
