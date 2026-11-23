#!/usr/bin/env python3
"""Phase 8 - frozen learning-curve subsets (experiment #5): nested 10 / 50 / 200 TRAIN images per class.

Per scheme-A class (person, helmet, head, harness) and scheme-B classes (scaffold_structure, scaffold_platform),
sample train images that contain the class (from build/<scheme>/train.coco.json). Subsets are nested
(10 is inside 50 is inside 200) and take whole duplicate groups only once. Relation subsets are made later,
from labeled queue images. Frozen with SHA256SUMS like the splits; a change means a new version.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path.home() / "GenIA-data"
OUT = ROOT / "splits" / "v1" / "learning_curve"
SIZES = [10, 50, 200]
SEED = 20261118


def main():
    if (OUT / "SHA256SUMS").exists():
        sys.exit("learning_curve v1 is frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    lines = []
    for scheme, classes in [("A_objects_v1", ["person", "helmet", "head", "harness"]),
                            ("B_objects_platform_v1", ["scaffold_structure", "scaffold_platform"])]:
        d = json.load(open(ROOT / "build" / scheme / "train.coco.json"))
        cat = {c["id"]: c["name"] for c in d["categories"]}
        has = {c: set() for c in classes}
        for a in d["annotations"]:
            if cat[a["category_id"]] in has:
                has[cat[a["category_id"]]].add(a["image_id"])
        gid = {im["id"]: (im["global_id"], im["group_id"]) for im in d["images"]}
        for c in classes:
            pool = sorted(has[c])
            rng.shuffle(pool)
            picked, groups = [], set()
            for i in pool:  # one image per group so tiny subsets are diverse
                g, grp = gid[i]
                if grp in groups:
                    continue
                groups.add(grp)
                picked.append(g)
                if len(picked) == max(SIZES):
                    break
            for n in SIZES:
                sub = picked[:n]
                (OUT / f"{c}_{n}.txt").write_text("".join(f"{g}\n" for g in sub))
                lines.append(f"{c}: {n} requested, {len(sub)} available")
    files = sorted(OUT.glob("*.txt"))
    (OUT / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in files))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
