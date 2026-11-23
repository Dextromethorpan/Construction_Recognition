"""Score a fine-tuned variant on the REAL relation-labeled test photos (Codex-reviewed labels, frozen_v1).
Per person (GT boxes): person recall, on-scaffold P/R/F1 (unclear GT excluded; unmatched person_on_scaffold
predictions count as FP), hard-negative FP rate on CS10k 'none' people, helmet and harness balanced accuracy.
Usage: evaluate.py WEIGHTS VARIANT_NAME [--conf 0.25]"""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import json, csv, sys, argparse
from pathlib import Path
from ultralytics import YOLO

D = Path(GENIA_DATA)
REL = D / "labels/relations/v1"
OUTDIR = D / "runs/ablation_eval"

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2]*a[3] + b[2]*b[3] - inter
    return inter / u if u > 0 else 0

def ioa(small, big):  # share of `small` inside `big`
    ax2, ay2, bx2, by2 = small[0]+small[2], small[1]+small[3], big[0]+big[2], big[1]+big[3]
    iw, ih = max(0, min(ax2, bx2) - max(small[0], big[0])), max(0, min(ay2, by2) - max(small[1], big[1]))
    return iw * ih / (small[2]*small[3]) if small[2]*small[3] > 0 else 0

def test_tasks():
    synth = {r["global_id"] for r in csv.DictReader(open(REL / "synthetic_images.csv"))}
    out = []
    for t in json.load(open(REL / "frozen_v1/codex_reviewed.json")):
        d = t["data"]
        if d.get("pass") != 1 or not t["annotations"] or not d["split"].startswith("test"): continue
        if d["global_id"] in synth: continue
        res = t["annotations"][-1]["result"]
        if any("bad_image" in c for r in res if r.get("from_name") == "image_flag" for c in r["value"]["choices"]): continue
        people = {}
        for r in res:
            v = r.get("value", {})
            if r.get("type") == "rectanglelabels" and v["rectanglelabels"][0] == "person":
                people.setdefault(r["id"], {})["box"] = (v["x"], v["y"], v["width"], v["height"])
            elif r.get("type") == "choices" and r.get("from_name") in ("support", "helmet_state", "harness_state"):
                people.setdefault(r["id"], {})[r["from_name"]] = v["choices"][0]
        img = D / d["image"].split("?d=")[1]
        out.append({"gid": d["global_id"], "split": d["split"], "img": str(img),
                    "people": [p for p in people.values() if "box" in p]})
    return out

def safe_div(a, b): return round(a / b, 4) if b else None

def score(tasks, preds, conf):
    S = {}
    for sub in ("all_real", "test_indist", "test_external", "test_heldout"):
        c = dict(ptp=0, pn=0, tp=0, fp=0, fn=0, spurious=0, hn=0, hn_fp=0,
                 h_pos=0, h_pos_ok=0, h_neg=0, h_neg_ok=0, z_pos=0, z_pos_ok=0, z_neg=0, z_neg_ok=0)
        for t in tasks:
            if sub != "all_real" and t["split"] != sub: continue
            P = [p for p in preds[t["gid"]] if p["conf"] >= conf]
            by = lambda k: [p["box"] for p in P if p["cls"] == k]
            persons, on, helm, harn, hw = by(0), by(6), by(1), by(3), by(8)
            for g in t["people"]:
                b = g["box"]; c["pn"] += 1
                c["ptp"] += any(iou(b, q) >= 0.5 for q in persons)
                pred_on = any(iou(b, q) >= 0.5 for q in on)
                s = g.get("support", "unclear")
                if s in ("stands_on", "climbing"): c["tp" if pred_on else "fn"] += 1
                elif s == "none":
                    c["fp"] += pred_on
                    if t["split"] == "test_external": c["hn"] += 1; c["hn_fp"] += pred_on
                head = (b[0], b[1], b[2], b[3] * 0.4)
                hpred = any(ioa(q, head) >= 0.5 for q in helm)
                hs = g.get("helmet_state")
                if hs == "helmet:worn": c["h_pos"] += 1; c["h_pos_ok"] += hpred
                elif hs == "helmet:not_worn": c["h_neg"] += 1; c["h_neg_ok"] += not hpred
                zpred = any(ioa(q, b) >= 0.5 for q in harn) or any(iou(b, q) >= 0.5 for q in hw)
                zs = g.get("harness_state")
                if zs == "harness:worn": c["z_pos"] += 1; c["z_pos_ok"] += zpred
                elif zs == "harness:not_worn": c["z_neg"] += 1; c["z_neg_ok"] += not zpred
            for q in on:
                if all(iou(g["box"], q) < 0.3 for g in t["people"]): c["spurious"] += 1
        fp_all = c["fp"] + c["spurious"]
        prec, rec = safe_div(c["tp"], c["tp"] + fp_all), safe_div(c["tp"], c["tp"] + c["fn"])
        f1 = round(2*prec*rec/(prec+rec), 4) if prec and rec else 0.0
        bal = lambda a, n, b, m: round((a/n + b/m)/2, 4) if n and m else None
        S[sub] = {"person_recall": safe_div(c["ptp"], c["pn"]), "on_P": prec, "on_R": rec, "on_F1": f1,
                  "on_support_pos": c["tp"] + c["fn"], "hardneg_fp_rate": safe_div(c["hn_fp"], c["hn"]), "hardneg_n": c["hn"],
                  "helmet_bal_acc": bal(c["h_pos_ok"], c["h_pos"], c["h_neg_ok"], c["h_neg"]), "helmet_n": [c["h_pos"], c["h_neg"]],
                  "harness_bal_acc": bal(c["z_pos_ok"], c["z_pos"], c["z_neg_ok"], c["z_neg"]), "harness_n": [c["z_pos"], c["z_neg"]],
                  "counts": c}
    return S

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("name")
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--imgsz", type=int, default=512)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tasks = test_tasks(); m = YOLO(a.weights); preds = {}
    for t in tasks:
        r = m.predict(t["img"], imgsz=a.imgsz, conf=0.05, device="mps", verbose=False, max_det=300)[0]
        W, H = r.orig_shape[1], r.orig_shape[0]
        preds[t["gid"]] = [{"cls": int(c), "conf": float(s), "box": (x1/W*100, y1/H*100, (x2-x1)/W*100, (y2-y1)/H*100)}
                           for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    json.dump(preds, open(OUTDIR / f"{a.name}_preds.json", "w"))
    S = score(tasks, preds, a.conf)
    json.dump({"variant": a.name, "weights": a.weights, "conf": a.conf, "n_images": len(tasks), "scores": S},
              open(OUTDIR / f"{a.name}_scores.json", "w"), indent=1)
    print(a.name, json.dumps(S["all_real"]))

if __name__ == "__main__":
    main()
