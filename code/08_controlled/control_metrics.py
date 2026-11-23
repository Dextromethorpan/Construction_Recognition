#!/usr/bin/env python3
"""Control metrics C1-C5 (Labeling v2 and controlled experiment setup §3.3). Reads frozen_v2 + judge.py outputs.

  control_metrics.py c1 --config CFG                  person coverage PROXY (all_people_boxed, box counts, unmatched
                                                      confident detections). True C1 needs Luciano's blind audit.
  control_metrics.py c2 [--people CSV]                κ per field (wraps 02_labeling/relation_v2/kappa_v2.py, pass 1 vs pass 2)
  control_metrics.py c3 --config CFG                  on vs not_on in train_v2, hard-negative share by support_surface
  control_metrics.py c4 --dir DIR --arm A             learning curve from scores_<A>_d<s>_f<frac>.csv (run judge.py --train-frac)
  control_metrics.py c5 --scores scores_<arm>_d<s>.csv    on-scaffold people: detector miss vs judge miss at FA 1.5% / 5%
"""
import argparse, collections, csv, glob, re, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import threshold as T

HOME = Path.home(); FROZEN = HOME / "GenIA-data/labels/relations/v2/frozen_v2"
KAPPA = HOME / "GenIA-code/02_labeling/relation_v2/kappa_v2.py"


def series_people(cfg_path):
    import judge as J
    cfg = J.load_cfg(cfg_path); return cfg, J.load_people(cfg)


def c1(a):
    import judge as J
    cfg, imgs = series_people(a.config)
    dets = J.detections(cfg, imgs) if J.resolve(cfg["detector"]["cache"]).exists() else {}
    apb = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(FROZEN / "people_v2.csv")):
        if r["pass"] == "1" and r["global_id"] in imgs: apb[imgs[r["global_id"]]["set"]][(r["global_id"], r["all_people_boxed"])] += 1
    print("C1 person coverage: PROXY only. The real C1 is a blind audit of 40-60 photos by Luciano (not done).")
    for s in ("train_v2", "val_v2", "cs10k_dev"):
        ims = [g for g, t in imgs.items() if t["set"] == s]
        no = {g for g, v in apb[s] if v == "no"}
        npeople = sum(len(imgs[g]["people"]) for g in ims)
        line = f"  {s}: {len(ims)} images, {npeople} labeled people, images marked all_people_boxed=no: {len(no)}"
        if dets:
            un = 0; conf_un = 0
            for g in ims:
                dp = [d for d in dets.get(g, []) if d["cls"] == cfg["detector"]["person_class"]]
                gt = [p["box"] for p in imgs[g]["people"]]
                for d in dp:
                    if all(J.iou(tuple(d["box"]), b) < 0.3 for b in gt):
                        un += 1; conf_un += d["conf"] >= 0.5
            line += f" | detections with no labeled person (IoU<0.3): {un} (conf>=0.5: {conf_un}) = possible unlabeled people"
        print(line)


def c2(a):
    people = a.people or str(FROZEN / "people_v2.csv")
    print("C2: pass 1 vs pass 2 on", people, "(official pre-review κ is frozen_v2/kappa_pass2_PRE_review_official.csv)")
    subprocess.run([sys.executable, str(KAPPA), people, "--pass2"] + (["--out", a.out] if a.out else []), check=True)


def c3(a):
    cfg, imgs = series_people(a.config)
    tr = [p for t in imgs.values() if t["set"] == "train_v2" for p in t["people"]]
    on = sum(p["on"] == "on" for p in tr); neg = [p for p in tr if p["on"] == "not_on"]
    print(f"C3 class balance, train_v2 ({cfg['data']['series']}): on {on}, not_on {len(neg)}, "
          f"dropped {sum(p['on'] not in ('on', 'not_on') for p in tr)} (unclear/unannotated); "
          f"negatives per positive {len(neg) / max(on, 1):.2f} (aim >= 1)")
    by = collections.Counter(p["support"] for p in neg)
    hard = by.get("other_elevated", 0)
    print(f"  not_on by support_surface: {dict(by)}; hard negatives (other_elevated, at height without scaffold): "
          f"{hard} = {hard / max(len(neg), 1):.1%}")
    near = sum(p["hn"].startswith("ground+") and "no_scaffold" not in p["hn"] for p in neg)
    print(f"  ground people near a scaffold (in_front_of/next_to/under/behind/far): {near} = {near / max(len(neg), 1):.1%}")
    rv = collections.Counter(p["review"] for p in tr if p["on"] in ("on", "not_on"))
    print(f"  usable train people by review_status: {dict(rv)}")


def c4(a):
    pts = collections.defaultdict(list)
    for f in glob.glob(str(Path(a.dir) / f"scores_{a.arm}_d*_f*.csv")) + glob.glob(str(Path(a.dir) / f"scores_{a.arm}_d*.csv")):
        m = re.search(r"_d(\d+)(?:_f([\d.]+))?\.csv$", f)
        if not m: continue
        frac = float(m.group(2)) if m.group(2) else float(a.default_frac)
        _, _, y, s = T.load(f); pts[frac].append(T.measure(y, s, T.BUDGETS[0])["recall"])
    print("C4 learning curve (recall at FA 1.5%, mean over draws). Flat = last doubling < 0.02 gain.")
    for fr in sorted(pts): print(f"  train fraction {fr:g}: {np.mean(pts[fr]):.3f} (n draws {len(pts[fr])})")


def c5(a):
    rows = [r for r in csv.DictReader(open(a.scores)) if r["on_scaffold"] in ("on", "not_on")]
    y = np.array([r["on_scaffold"] == "on" for r in rows]); s = np.array([float(r["score"]) for r in rows])
    matched = np.array([r.get("matched", "1") == "1" for r in rows])
    print(f"C5 error split, {a.scores}: {int(y.sum())} on-scaffold people")
    for b in T.BUDGETS:
        t = T.cutoff(y, s, b); hit = s >= t
        dm = int((y & ~matched).sum()); jm = int((y & matched & ~hit).sum()); tp = int((y & hit).sum())
        print(f"  FA {b:.1%} (t={t:.4f}): found {tp} | detector miss {dm} | judge miss {jm} "
              f"-> of {dm + jm} misses, detector {dm / max(dm + jm, 1):.0%}, judge {jm / max(dm + jm, 1):.0%}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("c1", "c3"): sub.add_parser(n).add_argument("--config", required=True)
    p = sub.add_parser("c2"); p.add_argument("--people"); p.add_argument("--out")
    p = sub.add_parser("c4"); p.add_argument("--dir", required=True); p.add_argument("--arm", required=True)
    p.add_argument("--default-frac", default=0.8)
    p = sub.add_parser("c5"); p.add_argument("--scores", required=True)
    a = ap.parse_args(); globals()[a.cmd](a)


if __name__ == "__main__":
    main()
