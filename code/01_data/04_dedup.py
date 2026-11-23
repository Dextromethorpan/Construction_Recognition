#!/usr/bin/env python3
"""Phase 4 - near-duplicate detection with perceptual hashes, within AND across sources.

1. pHash (imagehash, 64 bit) for every image in interim/*/annotations.coco.json.
   Cached in reports/phash.csv (idempotent: only new global ids are hashed).
2. Pairs with Hamming distance <= THRESHOLD are linked; connected components = clusters.
3. One keeper per cluster: the image whose source labels the most classes
   (richer labels), tie -> more boxes, tie -> global id order. The rest go to reports/duplicates.csv.
4. reports/clusters.csv maps every global id to its cluster id (used as part of group_id in Phase 5).
5. reports/hamming_hist.csv: distribution of the minimum distance per image (0..16) to check the threshold.

Usage: python 04_dedup.py [--threshold 6]
"""
import csv
import glob
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path.home() / "GenIA-data"
REPORTS = ROOT / "reports"
THRESHOLD = int(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 6


def load_images():
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "interim" / "*" / "annotations.coco.json"))):
        d = json.load(open(f))
        ncls = len({a["category_id"] for a in d["annotations"]}) if d["annotations"] else 0
        nbox = {}
        for a in d["annotations"]:
            nbox[a["image_id"]] = nbox.get(a["image_id"], 0) + 1
        for im in d["images"]:
            rows[im["global_id"]] = {"path": im["file_name"], "source": im["source"],
                                     "source_ncls": ncls, "nbox": nbox.get(im["id"], 0)}
    return rows


def hashes(rows):
    cache_f = REPORTS / "phash.csv"
    cache = {}
    if cache_f.exists():
        cache = {r["global_id"]: r["phash"] for r in csv.DictReader(cache_f.open())}
    todo = [g for g in rows if g not in cache]
    for g in tqdm(todo, desc="pHash"):
        with Image.open(rows[g]["path"]) as im:
            cache[g] = str(imagehash.phash(im.convert("RGB")))
    with cache_f.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "phash"])
        for g in sorted(cache):
            if g in rows:
                w.writerow([g, cache[g]])
    return cache


def popcount64(x):
    return np.bitwise_count(x) if hasattr(np, "bitwise_count") else np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)


def main():
    rows = load_images()
    ph = hashes(rows)
    ids = sorted(rows)
    H = np.array([int(ph[g], 16) for g in ids], dtype=np.uint64)
    n = len(ids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    min_d = np.full(n, 64, dtype=np.int64)
    pair_d = {}
    for i in tqdm(range(n), desc="pairs"):
        if i == n - 1:
            break
        d = popcount64(np.bitwise_xor(H[i + 1:], H[i])).astype(np.int64)
        if d.size:
            min_d[i] = min(min_d[i], d.min())
            np.minimum.at(min_d, np.arange(i + 1, n), d)
        for j in np.nonzero(d <= THRESHOLD)[0]:
            jj = i + 1 + int(j)
            pair_d[(i, jj)] = int(d[j])
            ri, rj = find(i), find(jj)
            if ri != rj:
                parent[ri] = rj
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    cid_of, dup_rows, cross = {}, [], 0
    for k, (root, members) in enumerate(sorted(clusters.items(), key=lambda kv: ids[min(kv[1])])):
        cid = f"dup{k:06d}"
        for m in members:
            cid_of[ids[m]] = cid
        if len(members) == 1:
            continue
        # Greedy keepers: clusters chain (A~B, B~C, but A !~ C), so one keeper per cluster would throw away
        # images that are not duplicates of it. Walk members richest-first; a member within THRESHOLD of an
        # existing keeper is a duplicate, otherwise it becomes a keeper. The whole cluster stays one split group.
        order = sorted(members, key=lambda m: (-rows[ids[m]]["source_ncls"], -rows[ids[m]]["nbox"], ids[m]))
        if len({rows[ids[m]]["source"] for m in members}) > 1:
            cross += 1
        keepers = []
        for m in order:
            dists = popcount64(np.bitwise_xor(H[keepers], H[m])) if keepers else np.array([])
            if len(dists) and dists.min() <= THRESHOLD:
                k = keepers[int(dists.argmin())]
                dup_rows.append({"cluster_id": cid, "global_id": ids[m], "kept": ids[k],
                                 "hamming_to_kept": int(dists.min()), "source": rows[ids[m]]["source"],
                                 "kept_source": rows[ids[k]]["source"]})
            else:
                keepers.append(m)
    with (REPORTS / "duplicates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "global_id", "kept", "hamming_to_kept", "source", "kept_source"])
        w.writeheader()
        w.writerows(dup_rows)
    with (REPORTS / "clusters.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "cluster_id", "is_duplicate"])
        dups = {r["global_id"] for r in dup_rows}
        for g in ids:
            w.writerow([g, cid_of[g], int(g in dups)])
    hist = np.bincount(np.minimum(min_d, 17), minlength=18)
    with (REPORTS / "hamming_hist.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["min_hamming", "n_images"])
        for d, c in enumerate(hist):
            w.writerow([f"{d}" if d < 17 else ">=17", int(c)])
    multi = sum(1 for m in clusters.values() if len(m) > 1)
    print(f"images {n}, threshold {THRESHOLD}, clusters with >1 image {multi} ({cross} cross-source), "
          f"duplicates removed {len(dup_rows)}")
    print("min-distance histogram:", {d: int(c) for d, c in enumerate(hist)})


if __name__ == "__main__":
    main()
