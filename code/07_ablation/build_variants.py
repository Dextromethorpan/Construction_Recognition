"""Build one YOLO fine-tune dataset per labeling-policy variant (ablation A).

Every variant = relation-labeled train/val images (labels set by the variant) + the same fixed replay
sample of PPE-only detector images (hardhat_wkrs, shwd, rf_harness_fypcv; seed 0). Scaffold sources are
kept out of the replay because their people carry no on-scaffold label.

Classes (same as base_data.yaml): 0 person 1 helmet 2 head 3 harness 4 scaffold_platform
5 scaffold_structure 6 person_on_scaffold 7 person_on_unclear 8 person_harness_worn

Variants
  V0 base     : Codex labels; unclear people masked out (grey fill, boxes removed); harness from boxes
  V1 unclear  : V0 but unclear people kept as class 7
  V2 harness  : V0 but harness = person box of people with harness_state worn (class 8), harness boxes dropped
  V3 claude   : V0 but people/support from Claude's pre-review export; helmet/head/harness boxes from the
                source dataset (Claude drew almost none)
  V4 nosynth  : V0 without the synthetic ESCO images
  V5 unclear_neg: V0 but unclear people kept as plain person (i.e. counted as not on scaffold)
"""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import json, csv, os, random, hashlib, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw

D = Path(GENIA_DATA)
REL = D / "labels/relations/v1"
CODEX = REL / "frozen_v1/codex_reviewed.json"
CLAUDE = REL / "export_20261118_220544.json"   # last export before any Codex write
B = Path(os.environ.get("ABL_B", D / "build/B_objects_platform_v1"))
OUT = Path(os.environ.get("ABL_OUT", D / "build/ablation_v1"))
NAMES = ["person", "helmet", "head", "harness", "scaffold_platform", "scaffold_structure",
         "person_on_scaffold", "person_on_unclear", "person_harness_worn"]
VARIANTS = os.environ.get("ABL_VARIANTS", "V0_base,V1_unclear_class,V2_harness_field,V3_claude,V4_no_synth,V5_unclear_neg").split(",")
REPLAY_SRC = ("hardhat_wkrs", "shwd", "rf_harness_fypcv")
N_REPLAY_TRAIN, N_REPLAY_VAL = 2000, 300

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def load(fn):
    out = {}
    for t in json.load(open(fn)):
        if t["data"].get("pass") != 1 or not t["annotations"]: continue
        if t["data"]["split"] not in ("train", "val"): continue
        out[t["data"]["global_id"]] = (t["data"]["split"], t["annotations"][-1]["result"])
    return out

def parse(result):
    """-> people {id: dict(box, support, helmet, harness)}, other [(cls_name, box)], flags"""
    people, other, flags = {}, [], set()
    for r in result:
        v = r.get("value", {})
        if r.get("from_name") == "image_flag": flags.update(v.get("choices", []))
        if r.get("type") == "rectanglelabels":
            box = (v["x"], v["y"], v["width"], v["height"])
            lab = v["rectanglelabels"][0]
            if lab == "person": people.setdefault(r["id"], {})["box"] = box
            elif lab in NAMES: other.append((lab, box))
        elif r.get("type") == "polygonlabels":
            xs = [p[0] for p in v["points"]]; ys = [p[1] for p in v["points"]]
            other.append(("scaffold_platform", (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))))
        elif r.get("type") == "choices" and r.get("from_name") in ("support", "helmet_state", "harness_state"):
            people.setdefault(r["id"], {})[r["from_name"]] = v["choices"][0]
    people = {k: p for k, p in people.items() if "box" in p}
    return people, other, flags

def yl(c, box):
    x, y, w, h = box
    cx, cy = (x + w / 2) / 100, (y + h / 2) / 100
    clamp = lambda a: min(max(a, 0.0), 1.0)
    return f"{NAMES.index(c)} {clamp(cx):.6f} {clamp(cy):.6f} {clamp(w/100):.6f} {clamp(h/100):.6f}"

def src_img(split, gid):
    return (B / split / "images" / (gid.replace("/", "__", 1) + ".jpg")).resolve()

def src_labels(split, gid, keep):
    f = B / split / "labels" / (gid.replace("/", "__", 1) + ".txt")
    return [l.strip() for l in open(f) if l.strip() and NAMES[int(l.split()[0])] in keep] if f.exists() else []

def main():
    codex, claude = load(CODEX), load(CLAUDE)
    synth = {r["global_id"] for r in csv.DictReader(open(REL / "synthetic_images.csv"))}
    rng = random.Random(0)
    replay = {}
    for split, n in (("train", N_REPLAY_TRAIN), ("val", N_REPLAY_VAL)):
        files = sorted(p.stem for p in (B / split / "images").iterdir()
                       if p.stem.split("__")[0] in REPLAY_SRC
                       and p.stem.replace("__", "/", 1) not in codex)
        replay[split] = rng.sample(files, n)
    stats = {}
    for var in VARIANTS:
        root = OUT / var
        if root.exists(): shutil.rmtree(root)
        st = {"train": {}, "val": {}}
        for split in ("train", "val"):
            (root / split / "images").mkdir(parents=True); (root / split / "labels").mkdir(parents=True)
            for stem in replay[split]:
                os.symlink((B / split / "images" / f"{stem}.jpg").resolve(), root / split / "images" / f"{stem}.jpg")
                shutil.copy(B / split / "labels" / f"{stem}.txt", root / split / "labels" / f"{stem}.txt")
            labsrc = claude if var == "V3_claude" else codex
            for gid, (sp, result) in sorted(labsrc.items()):
                if sp != split: continue
                if var == "V4_no_synth" and gid in synth: continue
                people, other, flags = parse(result)
                if any("bad_image" in f for f in flags): continue
                lines, mask = [], []
                if var == "V3_claude":
                    other = [o for o in other if o[0] not in ("helmet", "head", "harness")]
                    lines += src_labels(split, gid, {"helmet", "head", "harness"})
                for c, box in other:
                    if var == "V2_harness_field" and c == "harness": continue
                    lines.append(yl(c, box))
                for p in people.values():
                    s = p.get("support", "unclear")
                    if s == "unclear" and var in ("V0_base", "V2_harness_field", "V3_claude", "V4_no_synth"):
                        mask.append(p["box"]); continue
                    lines.append(yl("person", p["box"]))
                    if s in ("stands_on", "climbing"): lines.append(yl("person_on_scaffold", p["box"]))
                    elif s == "unclear" and var == "V1_unclear_class": lines.append(yl("person_on_unclear", p["box"]))
                    if var == "V2_harness_field" and p.get("harness_state") == "harness:worn":
                        lines.append(yl("person_harness_worn", p["box"]))
                    st[split][s] = st[split].get(s, 0) + 1
                stem = gid.replace("/", "__", 1)
                dst = root / split / "images" / f"{stem}.jpg"
                if mask:
                    im = Image.open(src_img(split, gid)).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
                    for x, y, w, h in mask: dr.rectangle([x/100*W, y/100*H, (x+w)/100*W, (y+h)/100*H], fill=(114, 114, 114))
                    im.save(dst, quality=95); st[split]["masked"] = st[split].get("masked", 0) + len(mask)
                else:
                    os.symlink(src_img(split, gid), dst)
                (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                st[split]["rel_images"] = st[split].get("rel_images", 0) + 1
        (root / "data.yaml").write_text(f"path: {root}\ntrain: train/images\nval: val/images\nnames:\n" +
                                        "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
        stats[var] = st
    manifest = {"codex_export": str(CODEX), "codex_sha256": sha(CODEX), "claude_export": str(CLAUDE),
                "claude_sha256": sha(CLAUDE), "replay_sources": REPLAY_SRC, "replay_seed": 0,
                "n_replay": [N_REPLAY_TRAIN, N_REPLAY_VAL], "stats": stats}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
