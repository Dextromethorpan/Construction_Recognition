#!/usr/bin/env python3
"""Render the 9 chosen val_v2 photos with demo.py's pipeline, a 3x3 grid and a README with truth vs output.
  .venv-train/bin/python 09_demo/make_presentation.py [--out ~/Documents/GenIA/Presentation/demo]"""
import argparse, csv, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "08_controlled"))
import demo, judge

PICKS = [  # (file stem, global_id, caption, why)
    ("01_helmet_success", "rf_scaffold_archikhu/hard_hat_workers253", "Helmet: success", "two hard hats, both found"),
    ("02_helmet_correct_no", "rf_scaffold_archikhu/49", "Helmet: correct 'no helmet'", "no one wears a helmet: 4 of 5 correct, 1 false helmet. Also a scaffold contrast: 2 on the deck flagged, 3 on the floor beside it not"),
    ("03_helmet_failure", "rf_scaffold_esco/R-scaffold-new-7", "Helmet: failure", "white helmet + face mask on a worker carrying a deck board: missed"),
    ("04_harness_success", "rf_scaffold_esco/R-scaffold-114", "Harness: worn at height", "full-body harness on a scaffold deck"),
    ("05_harness_correct_no", "rf_scaffold_archikhu/4", "Harness: correct 'no harness'", "four workers, none with a harness"),
    ("06_harness_failure", "rf_scaffold_archikhu/119", "Harness: failure", "harness worn by a small, distant worker at the top of a tower: missed"),
    ("07_scaffold_success", "rf_scaffold_esco/R-scaffold-new-30", "On scaffold: success", "three workers on decks, all flagged"),
    ("08_scaffold_correct_ground", "rf_scaffold_archikhu/601", "On scaffold: correct 'ground beside'", "man on the floor next to the mobile tower: not flagged (grey); man on the deck: flagged (green). Left man: support labelled unclear"),
    ("09_scaffold_failure", "rf_scaffold_archikhu/452", "On scaffold: failure", "worker climbing the tower frame gets no person box, so no alert"),
]
H = Path.home(); D = H / "GenIA-data"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(H / "Documents/GenIA/Presentation/demo")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = judge.load_cfg(H / "GenIA-code/experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
    man = {r["global_id"]: r for r in csv.DictReader(open(D / "splits/v2/manifest.csv"))}
    syn = {r["global_id"] for r in csv.DictReader(open(D / "labels/relations/v1/synthetic_images.csv"))}
    from ultralytics.utils import YAML
    lic = {d["id"]: d["license"] for d in YAML.load(str(D / "manifest.yaml"))["datasets"]}
    P = demo.Pipeline(); demo.CUT, demo.BACKEND = P.cut, P.backend; tiles = []; md = []
    for stem, gid, cap, why in PICKS:
        t = imgs[gid]; assert t["set"] == "val_v2" and not t["synthetic"] and gid not in syn and not gid.startswith("cs10k"), gid
        im = Image.open(t["img"]).convert("RGB"); res = P.run(im)
        demo.draw(im, res, f"{stem[:2]} · {cap}").save(out / f"{stem}.png"); tiles.append(out / f"{stem}.png")
        dets = [p["box"] for p in res["people"]]; m = judge.match([p["box"] for p in t["people"]], dets, 0.3)
        src = man[gid]["source"]; L = lic[src]
        md += [f"## {stem}.png — {cap}", "", f"- Source id: `{gid}` (split `val_v2`, real photo)", f"- File: `{t['img'].replace(str(H), '~')}`",
               f"- Licence: {L['name']} — {L.get('attribution')} ({L.get('url')})", f"- Why chosen: {why}",
               f"- Run: {len(res['people'])} person box(es), detector {res['det_ms']:.0f} ms, total {res['total_ms']:.0f} ms", "",
               "| person | truth: helmet / harness / support (on scaffold) | model: helmet / harness / scaffold |", "|---|---|---|"]
        for i, p in enumerate(t["people"]):
            j = m.get(i); q = res["people"][j] if j is not None else None
            mo = (f"{'Y' if q['helmet'] >= .5 else 'N'} {q['helmet']:.2f} / {'Y' if q['harness'] >= .5 else 'N'} {q['harness']:.2f} / "
                  f"{'Y' if q['on_yes'] else 'N'} {q['on']:.2f}") if q else "no box (detector miss)"
            md.append(f"| {p['pid']} | {p['helmet']} / {p['harness']} / {p['support']} ({p['on']}) | {mo} |")
        extra = [j for j in range(len(dets)) if j not in m.values()]
        for j in extra:
            q = res["people"][j]; md.append(f"| (extra box) | no labeled person | {q['helmet']:.2f} / {q['harness']:.2f} / {'Y' if q['on_yes'] else 'N'} {q['on']:.2f} |")
        md.append("")
    # 3x3 grid: each tile letterboxed to 900x900
    S = 900; grid = Image.new("RGB", (3 * S, 3 * S), (20, 20, 20))
    for k, f in enumerate(tiles):
        im = Image.open(f); s = min(S / im.width, S / im.height); im = im.resize((round(im.width * s), round(im.height * s)))
        grid.paste(im, ((k % 3) * S + (S - im.width) // 2, (k // 3) * S + (S - im.height) // 2))
    grid.save(out / "demo_grid.png")
    head = ["# GenIA demo — 9 photos", "",
            "Pipeline = the final tested one (experiments/20261123_final_test/test_final.py), draw d=0: detector D0 coreml_int8 "
            "(yolo11n, 512 px, conf 0.25, CPU) → per person: C1 context crop → on-scaffold head (on if prob ≥ cutoff "
            f"{P.cut:.3f} = val false-alarm budget 1.5%); C0 person crop → helmet / harness heads (worn if prob ≥ 0.5). "
            "Encoder: frozen yolo11n-cls. Heads: GenIA-code/09_demo/heads_d0.npz.", "",
            "Box colour: green = on scaffold, grey = not. Label: helmet · harness · scaffold, ✓/✗ + probability.", "",
            "All 9 photos are real photos from the val_v2 split (no CS10k, no synthetic images, not from the test set). "
            "Truth = frozen_v2 pass-1 labels. Values like `not_visible` / `unclear` are not scored in the study.", "",
            "Regenerate: `cd ~/GenIA-code && .venv-train/bin/python 09_demo/make_presentation.py`.", "",
            "Licences: all images come from Roboflow Universe datasets under CC BY 4.0 (attribution per photo below).", ""]
    (out / "README.md").write_text("\n".join(head + md)); print("wrote", out)


if __name__ == "__main__":
    main()
