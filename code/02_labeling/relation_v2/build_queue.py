#!/usr/bin/env python3
"""Build the relation v2 labeling queue from splits/v2/manifest.csv (PREPARE ONLY, never labels).

Outputs (labeling_queue/relation_v2/): queue.csv, label_studio_import.json.
Pre-annotations are editable SUGGESTIONS (Label Studio predictions), never saved as labels:
  relabel_v1  frozen v1 Codex-reviewed result: boxes, platform polygons, helmet_state, harness_state, ankles,
              image fields. v1 support / proximity / tied_off / visibility / feet_on are NOT copied (no anchoring);
              compare afterwards with parse_v2.py --frozen-v1.
  new         build/B_persons_v1 boxes (source boxes + COCO pseudo person boxes); platform boxes -> 4-point polygons.
  cs10k_dev   no boxes yet (CS10k has no person boxes); prefill_pose.py adds person boxes + ankles later.
Ankles from a pose model are added by prefill_pose.py (separate step, needs a model run).
Order: 1 relabel val, 2 relabel train, 3 new val, 4 cs10k_dev, 5 new train (scaffold sources), 6 new train
(rf_harness_main); seeded random order inside each; 15% pass-2 copies (stratified) at the end, blind,
with geometry-only suggestions (no prefilled helmet/harness, so kappa is not inflated).
test_v1 images are never queued.

Usage: GENIA_DATA=~/GenIA-data python3 build_queue.py
"""
import csv, json, os, random
from collections import Counter, defaultdict
from pathlib import Path

D = Path(os.environ.get("GENIA_DATA", Path.home() / "GenIA-data"))
OUT = D / "labeling_queue/relation_v2"
SEED = 20261120
DOUBLE_SHARE = 0.15
B_NAMES = {0: "person", 1: "helmet", 2: "head", 3: "harness", 5: "scaffold_structure"}   # 4 = platform -> polygon
KEEP_PERSON_FIELDS = {"helmet_state", "harness_state"}
KEEP_IMAGE_FIELDS = {"viewpoint", "distance", "lighting", "occlusion"}


def rel_path(file_name):
    return file_name.split("GenIA-data/", 1)[1]


def image_meta():
    meta = {}
    for src_dir in sorted((D / "interim").iterdir()):
        f = src_dir / "annotations.coco.json"
        if not f.exists(): continue
        for im in json.load(open(f))["images"]:
            meta[im["global_id"]] = im
    return meta


def pre_from_v1(task):
    """Frozen v1 result -> v2 suggestions (boxes, polygons, helmet, harness, ankles, image fields)."""
    res = task["annotations"][-1]["result"]
    out = []
    for r in res:
        fn, t = r.get("from_name"), r.get("type")
        if t == "relation":
            continue
        if fn in ("box", "platform"):
            out.append(r)
        elif t == "choices" and fn in KEEP_PERSON_FIELDS | KEEP_IMAGE_FIELDS:
            out.append(r)
        elif t == "keypointlabels" and r["value"]["keypointlabels"][0] in ("left_ankle", "right_ankle"):
            out.append(r)
    return out


def pre_from_b(gid, split, W, H):
    f = D / "build/B_persons_v1" / split / "labels" / (gid.replace("/", "__") + ".txt")
    out = []
    if not f.exists(): return out
    for k, line in enumerate(f.read_text().splitlines()):
        p = line.split()
        if len(p) != 5: continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        x, y = 100 * (cx - w / 2), 100 * (cy - h / 2)
        base = {"id": f"pre{k}", "to_name": "image", "original_width": W, "original_height": H}
        if c in B_NAMES:
            out.append({**base, "from_name": "box", "type": "rectanglelabels",
                        "value": {"x": x, "y": y, "width": 100 * w, "height": 100 * h, "rotation": 0,
                                  "rectanglelabels": [B_NAMES[c]]}})
        elif c == 4:
            pts = [[x, y], [x + 100 * w, y], [x + 100 * w, y + 100 * h], [x, y + 100 * h]]
            out.append({**base, "from_name": "platform", "type": "polygonlabels",
                        "value": {"points": pts, "closed": True, "polygonlabels": ["scaffold_platform"]}})
    return out


def priority(r):
    if r["task_kind"] == "relabel_v1":
        return 1 if r["split_v2"] == "val_v2" else 2
    if r["split_v2"] == "val_v2": return 3
    if r["split_v2"] == "cs10k_dev": return 4
    return 5 if r["source"] != "rf_harness_main" else 6


def main():
    rng = random.Random(SEED)
    man = [r for r in csv.DictReader(open(D / "splits/v2/manifest.csv")) if r["split_v2"] != "test_v1"]
    fz = {t["data"]["global_id"]: t for t in json.load(open(D / "labels/relations/v1/frozen_v1/codex_reviewed.json"))
          if t["data"].get("pass") == 1}
    meta = image_meta()
    for r in man:
        r["priority"] = priority(r)
    groups = defaultdict(list)
    for r in sorted(man, key=lambda r: r["global_id"]):
        groups[r["priority"]].append(r)
    ordered = []
    for p in sorted(groups):
        g = groups[p]; rng.shuffle(g); ordered += g
    # 15% double labeling, stratified by priority group
    for p, g in groups.items():
        for r in rng.sample(g, round(len(g) * DOUBLE_SHARE)):
            r["double_label"] = 1
    cols = ["order", "global_id", "split_v1", "split_v2", "source", "task_kind", "selected_by", "contrast_group",
            "site_or_sequence_id", "synthetic", "priority", "double_label", "reason"]
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    with open(OUT / "queue.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for i, r in enumerate(ordered, 1):
            im = meta[r["global_id"]]
            r.update(order=i, double_label=r.get("double_label", 0), site_or_sequence_id=im.get("group_id", r["global_id"]))
            w.writerow(r)
            W, H = im["width"], im["height"]
            if r["task_kind"] == "relabel_v1":
                pre, mv = pre_from_v1(fz[r["global_id"]]), "frozen_v1_codex_reviewed"
            elif r["split_v2"] == "cs10k_dev":
                pre, mv = [], None
            else:
                pre, mv = pre_from_b(r["global_id"], r["split_v1"], W, H), "B_persons_v1_source_plus_pseudo"
            for p in ([1, 2] if r["double_label"] else [1]):
                data = {"image": f"/data/local-files/?d={rel_path(im['file_name'])}", "global_id": r["global_id"],
                        "split": r["split_v1"], "split_v2": r["split_v2"], "task_kind": r["task_kind"],
                        "selected_by": r["selected_by"], "contrast_group": r["contrast_group"],
                        "site_or_sequence_id": r["site_or_sequence_id"], "synthetic": int(r["synthetic"]),
                        "priority": r["priority"], "pass": p,
                        "hint": f"v2 | {r['task_kind']} | {r['split_v2']} | {r['source']} | pass {p}"
                                + (" | SYNTHETIC" if r["synthetic"] == "1" else "")}
                # pass 2 gets geometry only (no prefilled choices), so kappa is not inflated by a shared suggestion
                pp = pre if p == 1 else [x for x in pre if x.get("from_name") in ("box", "platform")]
                tasks.append({"data": data, "predictions": [{"model_version": mv, "result": pp}] if pp else []})
    tasks.sort(key=lambda t: (t["data"]["pass"], t["data"]["priority"]))   # stable: keeps the shuffled order
    (OUT / "label_studio_import.json").write_text(json.dumps(tasks))
    c = Counter((t["data"]["pass"], t["data"]["priority"]) for t in tasks)
    print(f"queue: {len(ordered)} images, {len(tasks)} tasks | (pass, priority): {dict(sorted(c.items()))}")
    print(f"with pre-annotations: {sum(1 for t in tasks if t['predictions'])}")


if __name__ == "__main__":
    main()
