#!/usr/bin/env python3
"""Phase 5 - split by source and group, never by photo.

- test_external: ConstructionSite-10k (labeled by its authors; test / VLM eval only).
- test_heldout : one whole source held out for generalization (rf_harness_mks: people + harness + scaffolds).
- train / val / test_indist: remaining sources, ~70/15/15 of images, allocated by GROUP.
  group = duplicate cluster (reports/clusters.csv); a cluster is never split.
  If a cluster touches a held-out source, the whole cluster goes to that held-out split (leak guard).
  Stratified by (source, tag signature) so each split gets helmets, heads, harnesses, scaffolds.
- Only dedup keepers are listed. Tags below are SELECTION / REPORT hints from original classes, not labels.

Writes splits/<version>/{train,val,test_indist,test_heldout,test_external}.txt and reports/split_report.md.
Freezing (SHA256SUMS) is a separate step: python 05_split.py --freeze  (refuses if already frozen).
"""
import csv
import glob
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / "GenIA-data"
VERSION = "v1"
OUT = ROOT / "splits" / VERSION
EXTERNAL, HELDOUT = "cs10k", "rf_harness_mks"
SEED = 20261118
SPLITS = ["train", "val", "test_indist", "test_heldout", "test_external"]

TAGS = {  # original class -> report/stratification tag (never written as a label)
    "helmet": ["hardhat_wkrs:helmet", "shwd:hat", "rf_harness_main:helmet", "rf_scaffold_climb:worker_with_helmet",
               "rf_scaffold_climb:worker_with_helmet_climbing"],
    "head": ["hardhat_wkrs:head", "shwd:person", "rf_harness_main:No helmet", "rf_scaffold_climb:worker_without_helmet",
             "rf_scaffold_climb:worker_without_helmet_climbing"],
    "harness": ["rf_harness_main:harness", "rf_harness_mks:safety-harness", "rf_harness_fypcv:safety-harness"],
    "person": ["hardhat_wkrs:person", "rf_harness_main:person", "rf_harness_mks:human", "rf_harness_fypcv:human",
               "rf_scaffold_climb:worker_with_helmet", "rf_scaffold_climb:worker_without_helmet",
               "rf_scaffold_climb:worker_with_helmet_climbing", "rf_scaffold_climb:worker_without_helmet_climbing",
               "shwd:person", "shwd:hat", "hardhat_wkrs:head", "hardhat_wkrs:helmet"],
    "scaffold": ["rf_harness_mks:Suspension_scaffold", "rf_scaffold_platform:modular_scaffold",
                 "rf_scaffold_platform:prefabricated_frame_scaffold", "rf_scaffold_platform:tube_and_coupler_scaffold",
                 "rf_scaffold_platform:working_platform", "rf_scaffold_climb:mobile_scaffold_outtrigger",
                 "rf_scaffold_climb:mobile_scaffold_no_outtrigger"],
}
CLS2TAG = defaultdict(set)
for t, cs in TAGS.items():
    for c in cs:
        CLS2TAG[c].add(t)

# Generic rules for the scaffold sets added later (esco, archikhu, sparks, mobile, hook).
import re  # noqa: E402
GENERIC = [(r"worker|human|person", "person"), (r"scaffold", "scaffold"), (r"with_helmet|hardhat", "helmet"),
           (r"without_helmet", "head")]
for src in ["rf_scaffold_esco", "rf_scaffold_archikhu", "rf_scaffold_sparks", "rf_scaffold_mobile", "rf_scaffold_hook"]:
    CLS2TAG[src] = None  # marker only


def tags_for(cls_name):
    src = cls_name.split(":")[0]
    if src in CLS2TAG and CLS2TAG[src] is None:
        name = cls_name.split(":", 1)[1].lower()
        out = {t for rx, t in GENERIC if re.search(rx, name)}
        if "without_helmet" in name:
            out.discard("helmet")
        return out
    return CLS2TAG.get(cls_name, set())


def load():
    info = {}
    for f in sorted(glob.glob(str(ROOT / "interim" / "*" / "annotations.coco.json"))):
        d = json.load(open(f))
        cat = {c["id"]: c["name"] for c in d["categories"]}
        per = defaultdict(Counter)
        for a in d["annotations"]:
            per[a["image_id"]][cat[a["category_id"]]] += 1
        for im in d["images"]:
            cls = per[im["id"]]
            tags = set().union(*[tags_for(c) for c in cls]) if cls else set()
            if "scaffold" in tags and "person" not in tags:
                tags.add("empty_scaffold")
            info[im["global_id"]] = {"source": im["source"], "classes": cls, "tags": tags}
    return info


def main():
    if (OUT / "SHA256SUMS").exists():
        sys.exit(f"splits/{VERSION} is frozen; make a new version instead")
    if "--freeze" in sys.argv:
        files = [OUT / f"{s}.txt" for s in SPLITS]
        (OUT / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
        print(f"frozen splits/{VERSION}")
        return
    info = load()
    clusters, dup = defaultdict(list), set()
    for r in csv.DictReader(open(ROOT / "reports" / "clusters.csv")):
        clusters[r["cluster_id"]].append(r["global_id"])
        if r["is_duplicate"] == "1":
            dup.add(r["global_id"])
    assign, leak_guard = {}, []
    pool = []  # groups for train/val/test_indist
    for cid, members in clusters.items():
        srcs = {info[g]["source"] for g in members}
        if EXTERNAL in srcs or HELDOUT in srcs:
            split = "test_external" if EXTERNAL in srcs else "test_heldout"
            held = EXTERNAL if EXTERNAL in srcs else HELDOUT
            for g in members:
                if info[g]["source"] == held:
                    assign[g] = split
                else:
                    leak_guard.append((g, cid, split))
        else:
            pool.append((cid, members))
    # stratify: key = dominant source + sorted tag signature of the group's union of tags
    strata = defaultdict(list)
    for cid, members in pool:
        tags = set().union(*[info[g]["tags"] for g in members])
        src = Counter(info[g]["source"] for g in members).most_common(1)[0][0]
        strata[(src, tuple(sorted(tags & {"harness", "scaffold", "empty_scaffold", "head", "helmet"})))].append((cid, members))
    rng = random.Random(SEED)
    target = {"train": 0.70, "val": 0.15, "test_indist": 0.15}
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        groups.sort(key=lambda g: -len(g[1]))  # place big groups first, then fill
        tot = sum(len(m) for _, m in groups)
        got = Counter()
        for cid, members in groups:
            s = min(target, key=lambda s: (got[s] + len(members)) / (target[s] * tot) if tot else 0)
            got[s] += len(members)
            for g in members:
                assign[g] = s
    OUT.mkdir(parents=True, exist_ok=True)
    lists = defaultdict(list)
    for g, s in assign.items():
        if g not in dup:
            lists[s].append(g)
    for s in SPLITS:
        (OUT / f"{s}.txt").write_text("".join(f"{g}\n" for g in sorted(lists[s])))
    with open(ROOT / "reports" / "leak_guard_excluded.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "cluster_id", "excluded_because_cluster_in"])
        w.writerows(leak_guard)
    # checks
    split_of = {g: s for s, gs in lists.items() for g in gs}
    bad = [c for c, m in clusters.items() if len({split_of[g] for g in m if g in split_of}) > 1]
    report(info, lists, leak_guard, bad)


def report(info, lists, leak_guard, bad):
    srcs = sorted({v["source"] for v in info.values()})
    L = [f"# Split report, splits/{VERSION} (PROPOSED, not frozen)\n",
         f"Seed {SEED}. Held out: `{HELDOUT}` -> test_heldout, `{EXTERNAL}` -> test_external. "
         f"Groups = pHash duplicate clusters (threshold 6). Only dedup keepers are listed.\n",
         f"Check: clusters spanning two splits = **{len(bad)}**. Leak-guard exclusions: {len(leak_guard)}.\n",
         "## Images per split x source\n", "| split | " + " | ".join(srcs) + " | total |",
         "|---" * (len(srcs) + 2) + "|"]
    for s in SPLITS:
        c = Counter(info[g]["source"] for g in lists[s])
        L.append(f"| {s} | " + " | ".join(str(c.get(x, 0)) for x in srcs) + f" | **{sum(c.values())}** |")
    tags = ["person", "helmet", "head", "harness", "scaffold", "empty_scaffold"]
    L += ["\n## Images per split x tag (selection hints from original classes, not labels)\n",
          "| split | " + " | ".join(tags) + " |", "|---" * (len(tags) + 1) + "|"]
    for s in SPLITS:
        c = Counter(t for g in lists[s] for t in info[g]["tags"])
        L.append(f"| {s} | " + " | ".join(str(c.get(t, 0)) for t in tags) + " |")
    L += ["\n## Boxes per split x original class\n", "| class | " + " | ".join(SPLITS) + " |",
          "|---" * (len(SPLITS) + 1) + "|"]
    per = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for g in lists[s]:
            per[s].update(info[g]["classes"])
    for cls in sorted(set().union(*per.values())):
        L.append(f"| {cls} | " + " | ".join(str(per[s].get(cls, 0)) for s in SPLITS) + " |")
    rep = "\n".join(L) + "\n"
    (ROOT / "reports" / "split_report.md").write_text(rep)
    with open(ROOT / "reports" / "class_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class"] + SPLITS)
        for cls in sorted(set().union(*per.values())):
            w.writerow([cls] + [per[s].get(cls, 0) for s in SPLITS])
    print(rep)


if __name__ == "__main__":
    main()
