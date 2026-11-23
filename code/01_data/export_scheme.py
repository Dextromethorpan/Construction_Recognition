#!/usr/bin/env python3
"""Materialize a label scheme (A or B) into build/<scheme>/<split>/ (YOLO + image symlinks) and
build/<scheme>/<split>.coco.json. Builds are disposable; rebuilt from interim + scheme + frozen split.

Missing vs absent: every COCO image gets `not_exhaustive_category_ids` (LVIS convention): scheme classes
that its source does not label exhaustively. A loss should ignore those classes for that image.
YOLO cannot express this, so the YOLO export follows on_missing:
  not_exhaustive (default): keep the image, write only known boxes, and log how many images are affected
  exclude_image (--strict): drop images that have ANY non-exhaustive scheme class
Scheme C/D are human relation labels and are exported by the labeling pipeline, not here.

Usage: python export_scheme.py A_objects_v1 [--strict] [--splits train,val,test_indist,test_heldout]
"""
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path.home() / "GenIA-data"
SCHEMES = ROOT / "labels" / "schemes"


def resolve(name):
    s = yaml.safe_load((SCHEMES / f"{name}.yaml").read_text())
    if "extends" not in s:
        return s
    base = resolve(s["extends"])
    out = {**base, **{k: v for k, v in s.items() if k != "sources"}}
    srcs = {k: {"map": dict(v.get("map", {})), "exhaustive": list(v.get("exhaustive", []))}
            for k, v in base["sources"].items()}
    for k, v in s.get("sources", {}).items():
        cur = srcs.setdefault(k, {"map": {}, "exhaustive": []})
        if "map" in v:
            cur["map"].update(v["map"])
        cur["map"].update(v.get("add_map", {}))
        cur["exhaustive"] = sorted(set(cur["exhaustive"]) | set(v.get("exhaustive", [])) | set(v.get("add_exhaustive", [])))
    out["sources"] = srcs
    return out


def main(argv):
    name = argv[0]
    strict = "--strict" in argv
    splits = ["train", "val", "test_indist", "test_heldout"]
    if "--splits" in argv:
        splits = argv[argv.index("--splits") + 1].split(",")
    sch = resolve(name)
    classes = sch["classes"]
    cid = {c: i + 1 for i, c in enumerate(classes)}
    tag = name + ("_strict" if strict else "")
    out_root = ROOT / "build" / tag
    interim = {}
    for src in sch["sources"]:
        d = json.load(open(ROOT / "interim" / src / "annotations.coco.json"))
        cat = {c["id"]: c["name"].split(":", 1)[1] for c in d["categories"]}
        anns = {}
        for a in d["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        interim[src] = {im["global_id"]: (im, [(cat[a["category_id"]], a["bbox"]) for a in anns.get(im["id"], [])])
                        for im in d["images"]}
    counts, log = [], []
    for split in splits:
        gids = (ROOT / "splits" / "v1" / f"{split}.txt").read_text().split()
        coco = {"info": {"scheme": name, "split": split, "on_missing": "exclude_image" if strict else "not_exhaustive"},
                "categories": [{"id": cid[c], "name": c} for c in classes], "images": [], "annotations": []}
        ydir = out_root / split
        (ydir / "images").mkdir(parents=True, exist_ok=True)
        (ydir / "labels").mkdir(parents=True, exist_ok=True)
        c_box, c_img, n_partial, n_excluded, n_skipped_src = Counter(), Counter(), 0, 0, 0
        for gid in gids:
            src = gid.split("/")[0]
            if src not in sch["sources"] or src in sch.get("excluded_sources", []):
                n_skipped_src += 1
                continue
            im, boxes = interim[src][gid]
            m, exh = sch["sources"][src]["map"], set(sch["sources"][src]["exhaustive"])
            not_exh = [cid[c] for c in classes if c not in exh]
            if strict and not_exh:
                n_excluded += 1
                continue
            mapped = [(m[c], b) for c, b in boxes if c in m]
            if not mapped and not exh:
                n_excluded += 1  # source knows nothing about this scheme
                continue
            n_partial += bool(not_exh)
            iid = len(coco["images"]) + 1
            coco["images"].append({"id": iid, "global_id": gid, "file_name": im["file_name"], "width": im["width"],
                                   "height": im["height"], "source": src, "group_id": im["group_id"],
                                   "not_exhaustive_category_ids": not_exh})
            W, H = im["width"], im["height"]
            lines = []
            for c, (x, y, w, h) in mapped:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": iid, "category_id": cid[c],
                                            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                lines.append(f"{cid[c] - 1} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                c_box[c] += 1
            for c in {c for c, _ in mapped}:
                c_img[c] += 1
            stem = gid.replace("/", "__")
            ext = Path(im["file_name"]).suffix
            link = ydir / "images" / f"{stem}{ext}"
            if not link.exists():
                os.symlink(im["file_name"], link)
            (ydir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (out_root / f"{split}.coco.json").write_text(json.dumps(coco))
        for c in classes:
            counts.append([name, split, c, c_img[c], c_box[c]])
        log.append(f"{split}: {len(coco['images'])} images, partial-label images {n_partial}, "
                   f"excluded {n_excluded}, not in scheme {n_skipped_src}")
    (out_root / "data.yaml").write_text(yaml.safe_dump({"path": str(out_root), **{s: f"{s}/images" for s in splits},
                                                        "names": {i: c for i, c in enumerate(classes)}}, sort_keys=False))
    rep = ROOT / "reports" / f"class_counts_{tag}.csv"
    with rep.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "split", "class", "images", "boxes"])
        w.writerows(counts)
    (ROOT / "reports" / f"export_{tag}.log").write_text("\n".join(log) + "\n")
    print(f"== {tag}\n" + "\n".join(log))
    for r in counts:
        print("  ", *r)


if __name__ == "__main__":
    main(sys.argv[1:])
