#!/usr/bin/env python3
"""Build splits/v2 (relation v2 + controlled experiments). Reads only; never edits v1 files.

v2 is built INSIDE the frozen v1 split, so no image changes side:
  train_v2   v1 train relation images (relabel_v1) + NEW train images from scaffold sources
             (archikhu, sparks, platform, climb: all images with >= 1 person) + a seeded random sample of
             rf_harness_main train images with >= 1 person (at-height hard negatives).
  val_v2     same recipe on the v1 val split (threshold + winner selection).
  cs10k_dev  ~150 CS10k images from test_external, NOT in the v1 relation queue, not in the same pHash
             duplicate cluster as any of the 159 CS10k test images and with pHash Hamming > HAMMING_MIN to all of
             them. Stratified like the 159 test images (same caption categories as 07_queue.py).
  test_v1    the 336 real test photos exactly as 07_ablation/evaluate.py selects them (copied list). Opened once.
'People' for new images = person boxes in build/B_persons_v1 (source boxes + COCO pseudo boxes, conf >= 0.5).
rf_scaffold_mobile is not added (video near-duplicates; 30 train + 64 test images are already labeled).

Usage: GENIA_DATA=~/GenIA-data python3 build_splits_v2.py [--harness-train 300] [--harness-val 100] [--cs10k-dev 150]
Refuses to run if splits/v2/FROZEN exists.
"""
import argparse, csv, hashlib, json, os, random, re
from collections import Counter, defaultdict
from pathlib import Path

D = Path(os.environ.get("GENIA_DATA", Path.home() / "GenIA-data"))
V1 = D / "splits/v1"
OUT = D / "splits/v2"
REL = D / "labels/relations/v1"
SEED = 20261120
SCAFFOLD_SOURCES = ["rf_scaffold_archikhu", "rf_scaffold_sparks", "rf_scaffold_platform", "rf_scaffold_climb"]
HAMMING_MIN = 10          # dedup used 6; stricter here because cs10k_dev must not leak into the CS10k test


def split_of():
    out = {}
    for s in ["train", "val", "test_indist", "test_heldout", "test_external"]:
        for g in (V1 / f"{s}.txt").read_text().split():
            out[g] = s
    return out


def frozen_v1():
    """global_id -> pass-1 task (Codex-reviewed, frozen)."""
    out = {}
    for t in json.load(open(REL / "frozen_v1/codex_reviewed.json")):
        if t["data"].get("pass") == 1:
            out[t["data"]["global_id"]] = t
    return out


def is_bad(t):
    res = t["annotations"][-1]["result"] if t.get("annotations") else []
    return any("bad_image" in c for r in res if r.get("from_name") == "image_flag" for c in r["value"]["choices"])


def test_v1(fz, synth):
    """Same selection as 07_ablation/evaluate.py test_tasks()."""
    out = []
    for g, t in fz.items():
        d = t["data"]
        if not t["annotations"] or not d["split"].startswith("test") or g in synth or is_bad(t):
            continue
        out.append((g, d["split"]))
    return sorted(out)


def n_persons(gid, split):
    f = D / "build/B_persons_v1" / split / "labels" / (gid.replace("/", "__") + ".txt")
    if not f.exists():
        return None
    return sum(1 for line in f.read_text().splitlines() if line.split() and line.split()[0] == "0")


def cs10k_category(cap, violations):
    cap = (cap or "").lower(); has_sc = "scaffold" in cap
    if has_sc and re.search(r"worker|person|people|\bman\b|\bmen\b", cap):
        return "scaffold_workers"
    if (2 in (violations or []) and not has_sc) or (re.search(r"roof|edge|elevated|at height|crane|top of", cap) and not has_sc):
        return "at_height_no_scaffold"
    if not has_sc and re.search(r"ladder|formwork|fence|rebar", cap):
        return "lookalike"
    if re.search(r"\b(cap|caps|hood|hoodie|holding (a|his|her) (hard )?hat|helmet in (his|her) hand)", cap):
        return "cap_hood"
    return None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness-train", type=int, default=300)
    ap.add_argument("--harness-val", type=int, default=100)
    ap.add_argument("--cs10k-dev", type=int, default=150)
    a = ap.parse_args()
    assert not (OUT / "FROZEN").exists(), "splits/v2 is frozen; make splits/v3 instead"
    rng = random.Random(SEED)
    sp = split_of()
    fz = frozen_v1()
    synth = {r["global_id"] for r in csv.DictReader(open(REL / "synthetic_images.csv"))}
    v1_queue = {r["global_id"]: r for r in csv.DictReader(open(D / "labeling_queue/relation_v1/queue.csv"))}
    rows = []   # manifest rows

    def add(g, split_v2, kind, sel, reason, people=None, extra=None):
        rows.append(dict(global_id=g, split_v2=split_v2, split_v1=sp.get(g, ""), source=g.split("/")[0],
                         task_kind=kind, selected_by=sel, reason=reason, synthetic=int(g in synth),
                         people_prefill="" if people is None else people,
                         contrast_group=(v1_queue.get(g, {}).get("contrast_pair") or ""), **(extra or {})))

    # 1. test_v1 (copy of the evaluate.py selection)
    test = test_v1(fz, synth)
    for g, s in test:
        add(g, "test_v1", "frozen_test", "legacy_v1", f"v1 {s}")

    # 2. relabel v1 train / val relation images (incl. synthetic, flagged; experiments keep real_only and
    #    synthetic_fixed as separate series)
    for g, t in sorted(fz.items()):
        s = t["data"]["split"]
        if s in ("train", "val") and t["annotations"] and not is_bad(t):
            add(g, f"{s}_v2", "relabel_v1", "legacy_v1", f"v1 relation image ({s})")

    # 3. new images from the training / val reservoir (07 Q2)
    for s in ("train", "val"):
        for src in SCAFFOLD_SOURCES:
            pool = sorted(g for g, x in sp.items() if x == s and g.startswith(src + "/") and g not in v1_queue)
            for g in pool:
                n = n_persons(g, s)
                if n:
                    add(g, f"{s}_v2", "new", "random", f"{src}: every {s} image with >= 1 person", n)
        hm = sorted(g for g, x in sp.items() if x == s and g.startswith("rf_harness_main/") and g not in v1_queue)
        rng.shuffle(hm)
        k = a.harness_train if s == "train" else a.harness_val
        picked = 0
        for g in hm:
            if picked >= k: break
            n = n_persons(g, s)
            if n:
                add(g, f"{s}_v2", "new", "random", f"rf_harness_main: seeded random sample of {k} {s} images with people", n)
                picked += 1

    # 4. cs10k_dev
    ph = {r["global_id"]: int(r["phash"], 16) for r in csv.DictReader(open(D / "reports/phash.csv"))}
    cl = {r["global_id"]: r["cluster_id"] for r in csv.DictReader(open(D / "reports/clusters.csv"))}
    test_cs = [g for g, s in test if s == "test_external"]
    test_clusters = {cl.get(g) for g in test_cs}
    test_hash = [ph[g] for g in test_cs if g in ph]
    coco = json.load(open(D / "interim/cs10k/annotations.coco.json"))
    meta = {im["global_id"]: im for im in coco["images"]}
    test_cat = Counter(cs10k_category(meta[g].get("image_caption"), meta[g].get("violations")) for g in test_cs)
    by_cat = defaultdict(list)
    excluded = Counter()
    for g, s in sp.items():
        if s != "test_external" or g in v1_queue:
            continue
        cat = cs10k_category(meta[g].get("image_caption"), meta[g].get("violations")) if g in meta else None
        if cat is None:
            continue
        if cl.get(g) in test_clusters:
            excluded["same_dup_cluster_as_test"] += 1; continue
        if g not in ph:
            excluded["no_phash"] += 1; continue
        dmin = min(bin(ph[g] ^ h).count("1") for h in test_hash)
        if dmin <= HAMMING_MIN:
            excluded[f"hamming<={HAMMING_MIN}_to_test"] += 1; continue
        by_cat[cat].append((g, dmin))
    total_test = sum(v for k, v in test_cat.items() if k)
    quota = {c: round(a.cs10k_dev * n / total_test) for c, n in test_cat.items() if c}
    for c, q in sorted(quota.items()):
        pool = sorted(by_cat[c]); rng.shuffle(pool)
        for g, dmin in pool[:q]:
            add(g, "cs10k_dev", "new", "random", f"cs10k {c} (stratified like the 159 test images)", None,
                {"min_hamming_to_test": dmin})

    # 5. checks: no image in two v2 sets, v2 sets stay inside their v1 split
    seen = Counter(r["global_id"] for r in rows)
    dup = [g for g, n in seen.items() if n > 1]
    assert not dup, f"image in two v2 sets: {dup[:5]}"
    allowed = {"train_v2": {"train"}, "val_v2": {"val"}, "cs10k_dev": {"test_external"},
               "test_v1": {"test_indist", "test_external", "test_heldout"}}
    for r in rows:
        assert r["split_v1"] in allowed[r["split_v2"]], r

    # 6. write
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ("train_v2", "val_v2", "cs10k_dev", "test_v1"):
        (OUT / f"{s}.txt").write_text("\n".join(sorted(r["global_id"] for r in rows if r["split_v2"] == s)) + "\n")
    cols = ["global_id", "split_v2", "split_v1", "source", "task_kind", "selected_by", "reason", "synthetic",
            "people_prefill", "contrast_group", "min_hamming_to_test"]
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["split_v2"], r["global_id"])):
            w.writerow(r)
    # report
    c = Counter((r["split_v2"], r["task_kind"], r["source"], r["synthetic"]) for r in rows)
    ppl = Counter()
    for r in rows:
        if r["people_prefill"] != "": ppl[(r["split_v2"], r["source"])] += int(r["people_prefill"])
    lines = ["# splits/v2 build report", "", f"Seed {SEED}. Built by `GenIA-code/02_labeling/relation_v2/build_splits_v2.py`.", "",
             "| split_v2 | task_kind | source | synthetic | images | people (prefill, new only) |", "|---|---|---|---|---|---|"]
    for (s, k, src, syn), n in sorted(c.items()):
        lines.append(f"| {s} | {k} | {src} | {syn} | {n} | {ppl.get((s, src), '') if k == 'new' else ''} |")
    lines += ["", "| split_v2 | images |", "|---|---|"]
    for s, n in sorted(Counter(r["split_v2"] for r in rows).items()):
        lines.append(f"| {s} | {n} |")
    lines += ["", f"test_v1 by subset: {dict(Counter(s for _, s in test))} (expected 143 / 159 / 34 = 336)",
              f"CS10k test categories: {dict(test_cat)}", f"cs10k_dev quotas: {quota}",
              f"cs10k_dev candidates per category: { {k: len(v) for k, v in by_cat.items()} }",
              f"cs10k_dev exclusions: {dict(excluded)}"]
    (OUT / "build_report.md").write_text("\n".join(lines) + "\n")
    files = ["train_v2.txt", "val_v2.txt", "cs10k_dev.txt", "test_v1.txt", "manifest.csv"]
    (OUT / "SHA256SUMS").write_text("".join(f"{sha(OUT / f)}  {f}\n" for f in files))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
