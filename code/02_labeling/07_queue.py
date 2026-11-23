#!/usr/bin/env python3
"""Phase 7 - hand-labeling queue for the person-on-scaffold relation (PREPARE ONLY, never label).

Buckets are SELECTION HINTS computed from existing boxes, simple geometry and CS10k captions.
They are never written as labels.
  1 person_near_scaffold      person box overlaps/near a scaffold box, feet above the scaffold base (likely on)
  2 person_ground_near        person near a scaffold with feet at the scaffold base (hard negative: in front / under)
  3 empty_scaffold            scaffold, no person boxes, source labels persons exhaustively (VLM hallucination check)
  4 lookalike                 ladders, rebar, fences, formwork without scaffold boxes
  5 helmet_hard_negative      bare heads / caps / hoods / helmets in hand
  6 height_no_scaffold        people at height without a scaffold (roofs, edges, harness-at-height violations)
  7 contrast_pair             a bucket-1 image paired with a bucket-2 image from the same source and split
Priority: 1 = test_indist/test_heldout with person+scaffold (ALL of them), 2 = other test images,
          3 = val, 4 = train. rich_D = 1 for every test image and ~150 train images.
15% of the queue is double-labeled (second pass, blind) for Cohen's kappa.
Outputs: labeling_queue/relation_v1/{queue.csv, label_studio_import.json, label_studio_config.xml}
"""
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_data"))
from export_scheme import resolve  # noqa: E402

ROOT = Path.home() / "GenIA-data"
OUT = ROOT / "labeling_queue" / "relation_v1"
SEED = 20261118
TEST = {"test_indist", "test_heldout", "test_external"}
QUOTA = {2: 70, 3: 45, 4: 45, 5: 40, 6: 50}          # per bucket, on top of the mandatory priority-1 images
CS10K_SCAFFOLD_QUOTA = 60                              # test_external images whose caption says workers + scaffold
TRAIN_RICH = 150                                       # train images that get the rich D labels
CONTRAST_PAIRS = 30
DOUBLE_SHARE = 0.15
LOOKALIKE = {"vertical_ladder": "ladder", "ladder": "ladder", "rebar": "rebar"}


def iou_near(p, s, pad=0.1):
    px, py, pw, ph = p
    sx, sy, sw, sh = s
    sx, sy, sw, sh = sx - pad * sw, sy - pad * sh, sw * (1 + 2 * pad), sh * (1 + 2 * pad)
    return px < sx + sw and sx < px + pw and py < sy + sh and sy < py + ph


def main():
    sch = resolve("B_objects_platform_v1")
    split_of = {}
    for s in ["train", "val", "test_indist", "test_heldout", "test_external"]:
        for g in (ROOT / "splits" / "v1" / f"{s}.txt").read_text().split():
            split_of[g] = s
    rng = random.Random(SEED)
    cand = defaultdict(dict)   # bucket -> gid -> reason
    meta = {}
    for src_dir in sorted((ROOT / "interim").iterdir()):
        src = src_dir.name
        d = json.load(open(src_dir / "annotations.coco.json"))
        cat = {c["id"]: c["name"].split(":", 1)[1] for c in d["categories"]}
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        m = sch["sources"].get(src, {"map": {}, "exhaustive": []})
        for im in d["images"]:
            g = im["global_id"]
            if g not in split_of:
                continue  # dedup duplicates / leak-guard exclusions
            boxes = [(cat[a["category_id"]], a["bbox"]) for a in anns[im["id"]]]
            mapped = [(m["map"].get(c), b, c) for c, b in boxes]
            persons = [b for c, b, _ in mapped if c == "person"]
            scaff = [b for c, b, _ in mapped if c in ("scaffold_structure", "scaffold_platform")]
            meta[g] = {"im": im, "boxes": [(c, b, o) for c, b, o in mapped], "split": split_of[g], "source": src}
            if src == "cs10k":
                cap = (im.get("image_caption") or "").lower()
                has_sc = "scaffold" in cap
                if has_sc and re.search(r"worker|person|people|\bman\b|\bmen\b", cap):
                    cand["cs10k_scaffold"][g] = "caption mentions workers and scaffold"
                if 2 in (im.get("violations") or []) and not has_sc:
                    cand[6][g] = "CS10k rule 2 violation (harness at height) without scaffold in caption"
                elif re.search(r"roof|edge|elevated|at height|crane|top of", cap) and not has_sc:
                    cand[6][g] = "caption: at height, no scaffold"
                if re.search(r"\b(cap|caps|hood|hoodie|holding (a|his|her) (hard )?hat|helmet in (his|her) hand)", cap):
                    cand[5][g] = "caption: cap / hood / helmet in hand"
                if not has_sc and re.search(r"ladder|formwork|fence|rebar", cap):
                    cand[4][g] = "caption: look-alike structure, no scaffold"
                continue
            near_ground, near_up = False, False
            for p in persons:
                for s in scaff:
                    if iou_near(p, s):
                        feet = p[1] + p[3]
                        base = s[1] + s[3]
                        if feet >= base - 0.1 * s[3]:
                            near_ground = True
                        else:
                            near_up = True
            if near_up:
                cand[1][g] = "person box near scaffold, feet above scaffold base"
            if near_ground:
                cand[2][g] = "person near scaffold, feet at scaffold base"
            if scaff and not persons and "person" in m["exhaustive"]:
                cand[3][g] = "scaffold, no person (person labeled exhaustively)"
            elif scaff and not persons and src == "rf_harness_mks":
                cand[3][g] = "scaffold, no person box (person NOT exhaustive here: verify)"
            look = sorted({LOOKALIKE[o] for _, _, o in mapped if o in LOOKALIKE})
            if look and not scaff:
                cand[4][g] = "look-alike: " + ",".join(look)
            if any(o in ("No helmet", "head") for _, _, o in mapped) and persons:
                cand[5][g] = "bare heads next to people (cap/hood check)"
            if any(o in ("lifeline", "anchored", "NotAnchored") for _, _, o in mapped) and not scaff:
                cand[6][g] = "lifeline/anchor boxes without scaffold (work at height)"
    queue = {}

    def add(g, bucket, reason, prio):
        if g in queue:
            queue[g]["bucket"] += f"|{bucket}"
            return
        s = meta[g]["split"]
        queue[g] = {"global_id": g, "split": s, "source": meta[g]["source"], "bucket": str(bucket),
                    "reason": reason, "priority": prio}

    def prio_of(g):
        return 2 if meta[g]["split"] in TEST else (3 if meta[g]["split"] == "val" else 4)

    # priority 1: every test_indist / test_heldout image with person + scaffold
    for b in (1, 2):
        for g, r in cand[b].items():
            if meta[g]["split"] in ("test_indist", "test_heldout"):
                add(g, b, r, 1)
    # CS10k scaffold captions (test_external)
    cs = sorted(cand["cs10k_scaffold"])
    rng.shuffle(cs)
    for g in cs[:CS10K_SCAFFOLD_QUOTA]:
        add(g, 1, cand["cs10k_scaffold"][g], 2)
    # other buckets: test first, then val, then train
    for b, q in QUOTA.items():
        pool = sorted(g for g in cand[b] if g not in queue)
        rng.shuffle(pool)
        pool.sort(key=prio_of)
        for g in pool[:q]:
            add(g, b, cand[b][g], prio_of(g))
    # train images for relation training + rich D (bucket 1/2 from train)
    tr = sorted(g for b in (1, 2) for g in cand[b] if meta[g]["split"] == "train" and g not in queue)
    rng.shuffle(tr)
    for g in tr[:TRAIN_RICH]:
        add(g, 1 if g in cand[1] else 2, cand[1].get(g) or cand[2][g], 4)
    # contrast pairs: same source + split, one bucket-1 and one bucket-2 image
    by_key = defaultdict(lambda: ([], []))
    for g, q in queue.items():
        if q["bucket"].startswith("1") and g in cand[1] and g not in cand[2]:
            by_key[(q["source"], q["split"])][0].append(g)
        if q["bucket"].startswith("2") and g in cand[2] and g not in cand[1]:
            by_key[(q["source"], q["split"])][1].append(g)
    pid = 0
    for key in sorted(by_key):
        ons, offs = by_key[key]
        rng.shuffle(ons)
        rng.shuffle(offs)
        for a, b in zip(ons, offs):
            if pid >= CONTRAST_PAIRS:
                break
            pid += 1
            for g in (a, b):
                queue[g]["bucket"] += "|7"
                queue[g]["contrast_pair"] = f"cp{pid:03d}"
    # rich D: every test image + the train relation images
    for q in queue.values():
        q["rich_D"] = int(q["split"] in TEST or q["priority"] == 4)
    # 15% double labeling, stratified by first bucket
    by_b = defaultdict(list)
    for g, q in sorted(queue.items()):
        by_b[q["bucket"].split("|")[0]].append(g)
    for b, gs in by_b.items():
        rng.shuffle(gs)
        for g in gs[:round(len(gs) * DOUBLE_SHARE)]:
            queue[g]["double_label"] = 1
    rows = sorted(queue.values(), key=lambda q: (q["priority"], q["bucket"], q["global_id"]))
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["order", "global_id", "split", "source", "bucket", "reason", "priority", "rich_D", "double_label", "contrast_pair"]
    with (OUT / "queue.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, q in enumerate(rows, 1):
            w.writerow({"order": i, "double_label": q.get("double_label", 0), "contrast_pair": q.get("contrast_pair", ""),
                        **{k: q[k] for k in fields if k in q and k not in ("double_label", "contrast_pair")}})
    write_label_studio(rows, meta)
    c = Counter(b for q in rows for b in q["bucket"].split("|"))
    print(f"queue: {len(rows)} images | by bucket {dict(sorted(c.items()))} | by priority "
          f"{dict(sorted(Counter(q['priority'] for q in rows).items()))} | by split {dict(Counter(q['split'] for q in rows))} "
          f"| rich_D {sum(q['rich_D'] for q in rows)} | double {sum(q.get('double_label', 0) for q in rows)} | pairs {pid}")
    print("candidates per bucket:", {k: len(v) for k, v in cand.items()})


PRE = {"person": "person", "helmet": "helmet", "head": "head", "harness": "harness",
       "scaffold_structure": "scaffold_structure", "scaffold_platform": "scaffold_platform"}


def write_label_studio(rows, meta):
    """Tasks with existing boxes as editable PRE-ANNOTATIONS (predictions), never as final labels."""
    tasks = []
    for q in rows:
        g = q["global_id"]
        im = meta[g]["im"]
        W, H = im["width"], im["height"]
        rel = Path(im["file_name"]).relative_to(ROOT)
        res = []
        for k, (c, (x, y, w, h), orig) in enumerate(meta[g]["boxes"]):
            lab = PRE.get(c) or ({"vertical_ladder": "ladder", "ladder": "ladder", "rebar": "rebar"}.get(orig))
            if not lab:
                continue
            res.append({"id": f"pre{k}", "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                        "original_width": W, "original_height": H,
                        "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                                  "rotation": 0, "rectanglelabels": [lab]}})
        passes = [1, 2] if q.get("double_label") else [1]
        for p in passes:
            tasks.append({"data": {"image": f"/data/local-files/?d={rel}", "global_id": g, "split": q["split"],
                                   "bucket": q["bucket"], "priority": q["priority"], "rich_D": q["rich_D"],
                                   "pass": p, "hint": f"bucket {q['bucket']} | {q['split']} | "
                                   f"{'RICH (fill D fields)' if q['rich_D'] else 'basic (C only)'} | pass {p}"},
                          "predictions": [{"model_version": "existing_dataset_boxes_v1", "result": res}] if res else []})
    # second passes go to the end so they are labeled blind, after the first pass
    tasks.sort(key=lambda t: (t["data"]["pass"], t["data"]["priority"]))
    (OUT / "label_studio_import.json").write_text(json.dumps(tasks))
    print(f"label studio tasks: {len(tasks)}")


if __name__ == "__main__":
    main()
