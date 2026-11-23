#!/usr/bin/env python3
"""Phase 3 - convert every downloaded dataset to ONE COCO file per source, keeping ORIGINAL labels.

Output: interim/<id>/annotations.coco.json
- categories keep original names, prefixed with the source ("shwd:hat").
- images reference files in raw/ (or extracted/ for archived sources), never copied.
- global id = <dataset_id>/<original_filename_stem>.
- per-image metadata: source, original_split, width, height, group_id.
- validation: readable image, boxes clipped-inside check, no zero-area boxes.
  Broken images are dropped and logged in reports/dropped_images.csv.

Roboflow augmented exports: copies of one source photo share the stem before "_<ext>.rf.<hash>".
Only ONE copy per stem is kept (unaugmented valid/test copy if any, else first sorted);
the others are logged as dropped with reason 'augmented_copy'. This keeps global ids unique.

Idempotent: a source whose output exists is skipped unless --force.
Usage: python 03_convert.py [dataset_id ...] [--force]
"""
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path.home() / "GenIA-data"
RAW, EXTRACTED, INTERIM, REPORTS = ROOT / "raw", ROOT / "extracted", ROOT / "interim", ROOT / "reports"
ROBOFLOW_SETS = ["hardhat_wkrs", "rf_harness_main", "rf_harness_mks", "rf_harness_fypcv",
                 "rf_scaffold_platform", "rf_scaffold_climb", "rf_scaffold_esco", "rf_scaffold_archikhu",
                 "rf_scaffold_sparks", "rf_scaffold_mobile", "rf_scaffold_hook"]
RF_STEM = re.compile(r"^(?P<stem>.+?)_(?P<ext>jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]+\.\w+$", re.I)
SPLIT_PREF = {"test": 0, "valid": 1, "train": 2}

dropped = []  # rows: global_id, path, reason


def drop(gid, path, reason):
    dropped.append({"global_id": gid, "path": str(path), "reason": reason})


def read_size(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:120]


def valid_box(b, w, h):
    x, y, bw, bh = b
    if bw <= 0 or bh <= 0:
        return None, "zero_area"
    # clip boxes that spill over by rounding; reject boxes that are mostly outside
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
    if x1 - x0 <= 1 or y1 - y0 <= 1:
        return None, "outside_image"
    return [x0, y0, x1 - x0, y1 - y0], None


class Builder:
    def __init__(self, ds_id):
        self.ds, self.images, self.anns, self.cats = ds_id, [], [], {}
        self.box_issues = {"zero_area": 0, "outside_image": 0}

    def cat(self, name):
        key = f"{self.ds}:{name}"
        if key not in self.cats:
            self.cats[key] = len(self.cats) + 1
        return self.cats[key]

    def add_image(self, gid, path, split, group_id=None, extra=None):
        size = read_size(path)
        if size is None or (isinstance(size, tuple) and isinstance(size[1], str)):
            drop(gid, path, f"unreadable: {size[1] if size else ''}")
            return None
        w, h = size
        img = {"id": len(self.images) + 1, "global_id": gid, "file_name": str(path), "width": w, "height": h,
               "source": self.ds, "original_split": split, "group_id": group_id or gid}
        if extra:
            img.update(extra)
        self.images.append(img)
        return img

    def add_ann(self, img, cat_name, bbox, extra=None):
        box, err = valid_box(bbox, img["width"], img["height"])
        if err:
            self.box_issues[err] += 1
            return
        a = {"id": len(self.anns) + 1, "image_id": img["id"], "category_id": self.cat(cat_name),
             "bbox": [round(v, 2) for v in box], "area": round(box[2] * box[3], 2), "iscrowd": 0}
        if extra:
            a.update(extra)
        self.anns.append(a)

    def save(self, notes=""):
        out = INTERIM / self.ds
        out.mkdir(parents=True, exist_ok=True)
        coco = {"info": {"source": self.ds, "notes": notes, "box_issues_dropped": self.box_issues},
                "images": self.images, "annotations": self.anns,
                "categories": [{"id": i, "name": n, "supercategory": self.ds} for n, i in self.cats.items()]}
        (out / "annotations.coco.json").write_text(json.dumps(coco))
        print(f"{self.ds}: {len(self.images)} images, {len(self.anns)} boxes, {len(self.cats)} classes, "
              f"box issues {self.box_issues}")


def convert_roboflow(ds):
    b = Builder(ds)
    entries = []  # (stem, split_pref, file, split, coco_img, anns, catmap)
    for split_dir in sorted(p for p in (RAW / ds).iterdir() if p.is_dir()):
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        catmap = {c["id"]: c["name"] for c in coco["categories"]}
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            fn = im["file_name"]
            m = RF_STEM.match(fn)
            stem = m.group("stem") if m else Path(fn).stem
            entries.append((stem, SPLIT_PREF.get(split_dir.name, 3), fn, split_dir.name, split_dir,
                            by_img.get(im["id"], []), catmap))
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    seen = set()
    for stem, _, fn, split, sdir, anns, catmap in entries:
        gid = f"{ds}/{stem}"
        if gid in seen:
            drop(gid, sdir / fn, "augmented_copy")
            continue
        seen.add(gid)
        img = b.add_image(gid, sdir / fn, split)
        if img is None:
            continue
        for a in anns:
            b.add_ann(img, catmap[a["category_id"]], a["bbox"])
    b.save("Roboflow COCO export; one copy kept per original stem")


def convert_shwd():
    ds = "shwd"
    zpath = RAW / ds / "VOC2028.zip"
    ext = EXTRACTED / ds
    if not (ext / "VOC2028").exists():
        ext.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():  # guard against path traversal
                if n.startswith("/") or ".." in Path(n).parts:
                    raise SystemExit(f"unsafe path in zip: {n}")
            z.extractall(ext)
    voc = ext / "VOC2028"
    split_of = {}
    for s in ("train", "val", "test"):
        f = voc / "ImageSets" / "Main" / f"{s}.txt"
        if f.exists():
            for line in f.read_text().split():
                split_of[line.strip()] = s
    b = Builder(ds)
    for xml in sorted((voc / "Annotations").glob("*.xml")):
        stem = xml.stem
        jp = voc / "JPEGImages" / f"{stem}.jpg"
        if not jp.exists():
            drop(f"{ds}/{stem}", jp, "missing_image")
            continue
        img = b.add_image(f"{ds}/{stem}", jp, split_of.get(stem, "unknown"))
        if img is None:
            continue
        root = ET.parse(xml).getroot()
        for obj in root.iter("object"):
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            b.add_ann(img, obj.find("name").text.strip(), [x0, y0, x1 - x0, y1 - y0])
    b.save("Pascal VOC; 'person' = head without helmet")


def convert_cs10k():
    """ConstructionSite-10k parquet -> JPEGs in extracted/cs10k/ + COCO. Boxes are normalized xyxy.
    Rule violations are exhaustive per image (no violation = none). Image attributes and caption are kept."""
    import pyarrow.parquet as pq
    ds = "cs10k"
    ext = EXTRACTED / ds
    ext.mkdir(parents=True, exist_ok=True)
    b = Builder(ds)
    files = [("test", RAW / ds / "test.parquet"), ("train", RAW / ds / "train-00001-of-00002.parquet"),
             ("train", RAW / ds / "train-00002-of-00002.parquet")]
    attrs = ["image_caption", "illumination", "camera_distance", "view", "quality_of_info"]
    for split, f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):  # stream, never the whole file in memory
            for r in pf.read_row_group(rg).to_pylist():
                iid = r["image_id"]
                jp = ext / f"{iid}.jpg"
                if not jp.exists():
                    jp.write_bytes(r["image"]["bytes"])
                img = b.add_image(f"{ds}/{iid}", jp, split, extra={k: r.get(k) for k in attrs} | {
                    "violations": [n for n in range(1, 5) if r.get(f"rule_{n}_violation")]})
                if img is None:
                    continue
                W, H = img["width"], img["height"]

                def add(name, boxes, extra=None):
                    for x0, y0, x1, y1 in boxes or []:
                        b.add_ann(img, name, [x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H], extra)
                for n in range(1, 5):
                    v = r.get(f"rule_{n}_violation")
                    if v:
                        add(f"rule_{n}_violation", v.get("bounding_box"), {"reason": v.get("reason")})
                for obj in ("excavator", "rebar", "worker_with_white_hard_hat"):
                    add(obj, r.get(obj))
    b.save("Parquet; normalized xyxy boxes converted to pixel xywh; test/VLM evaluation only, never training")


def main(argv):
    force = "--force" in argv
    targets = [a for a in argv if not a.startswith("--")] or ROBOFLOW_SETS + ["shwd"]
    for ds in targets:
        if (INTERIM / ds / "annotations.coco.json").exists() and not force:
            print(f"{ds}: already converted, skipping")
            continue
        if ds in ROBOFLOW_SETS:
            convert_roboflow(ds)
        elif ds == "shwd":
            convert_shwd()
        elif ds == "cs10k":
            convert_cs10k()
        else:
            print(f"{ds}: no converter yet")
    REPORTS.mkdir(exist_ok=True)
    rep = REPORTS / "dropped_images.csv"
    old = list(csv.DictReader(rep.open())) if rep.exists() else []
    done_sources = {d["global_id"].split("/")[0] for d in dropped}
    rows = [r for r in old if r["global_id"].split("/")[0] not in done_sources] + dropped
    with rep.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["global_id", "path", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(f"dropped this run: {len(dropped)} -> {rep}")


if __name__ == "__main__":
    main(sys.argv[1:])
