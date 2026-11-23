"""Step 2: two-stage pipeline. A detector finds people; the per-person classifier from design B judges them.
Classifier: trained ONLY on train+val relation images (Codex labels, recommended policy: drop unclear,
helmet and harness from the per-person fields), GT boxes as training input. No test image is used for training.
Test: the same 336 real photos / 1,078 people as design A.
 - on-scaffold: GT person matched to a detection (IoU>=0.5) judged 'on'; undetected = not flagged (like A);
   detections judged 'on' that hit no GT person (IoU<0.3) are false positives.
 - helmet / harness: balanced accuracy on the GT people the detector found (reported with person recall).
Usage: combined.py PERSON_WEIGHTS NAME [--conf 0.25]"""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import sys, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
sys.path.insert(0, str(Path(__file__).parent))
import design_B as B
import evaluate as E

ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("name")
ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--neg", type=int, default=0); ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()
REL = B.REL; OUT = Path(GENIA_DATA + "/runs/combined"); OUT.mkdir(parents=True, exist_ok=True)
emb = YOLO("yolo11n-cls.pt")

def feats(img_path, boxes):
    im = Image.open(img_path).convert("RGB"); W, H = im.size; crops, geo = [], []
    for x, y, w, h in boxes:
        cx, cy = x + w/2, y + h/2
        x1, x2 = max(0, cx - w), min(100, cx + w); y1, y2 = max(0, cy - 0.75*h), min(100, cy + 0.75*h)
        crops.append(im.crop((x1/100*W, y1/100*H, max(x2/100*W, x1/100*W+2), max(y2/100*H, y1/100*H+2))).resize((224, 224)))
        geo.append([cx/100, cy/100, w/100, h/100, (y+h)/100, len(boxes)/10])
    if not crops: return np.zeros((0, 1))
    return np.array([np.concatenate([e.cpu().numpy(), g]) for e, g in zip(emb.embed(crops, imgsz=224, verbose=False), geo)], dtype=np.float32)

# --- train the three classifiers on train+val only
codex = B.load(REL / "frozen_v1/codex_reviewed.json")
X, keys = B.featurize(codex, emb, "codex"); row = {tuple(k): i for i, k in enumerate(keys)}
trg = [g for g, t in codex.items() if t["split"] in ("train", "val")]
TARGETS = {"on": lambda p: B.ON.get(p.get("support")), "helmet": lambda p: B.HELMET.get(p.get("helmet_state")),
           "harness": lambda p: {"harness:worn": 1, "harness:not_worn": 0}.get(p.get("harness_state"))}
TR = {}
for k, f in TARGETS.items():
    rows = [(X[row[(g, i)]], f(p)) for g in trg for i, p in enumerate(codex[g]["people"]) if f(p) is not None]
    TR[k] = (np.stack([x for x, _ in rows]), np.array([y for _, y in rows]))
    print(k, "train people:", len(rows), "positives:", int(TR[k][1].sum()), flush=True)

# --- optional extra 'not on scaffold' examples: people in ground-level PPE photos (hardhat_wkrs, shwd train),
# boxes from the completed person labels (B_persons_v1). Assumption: these close-up helmet photos are not on scaffolds.
if a.neg:
    import random
    PB = Path(GENIA_DATA + "/build/B_persons_v1/train")
    stems = sorted(p.stem for p in (PB/"labels").iterdir() if p.stem.split("__")[0] in ("hardhat_wkrs", "shwd"))
    random.Random(0).shuffle(stems); negF = []
    for st in stems:
        bx = [tuple(map(float, l.split()[1:])) for l in open(PB/"labels"/f"{st}.txt") if l.startswith("0 ")]
        bx = [((cx-w/2)*100, (cy-h/2)*100, w*100, h*100) for cx, cy, w, h in bx if w*h > 0.002]
        if bx: negF.append(feats(str(PB/"images"/f"{st}.jpg"), bx[:3]))
        if sum(len(f) for f in negF) >= a.neg: break
    negF = np.concatenate(negF)[:a.neg]
    TR["on"] = (np.concatenate([TR["on"][0], negF]), np.concatenate([TR["on"][1], np.zeros(len(negF), dtype=int)]))
    print("added negatives:", len(negF), flush=True)

# --- detect people on the test photos, judge them
det = YOLO(a.weights); tasks = E.test_tasks()
c = dict(pn=0, ptp=0, tp=0, fp=0, fn=0, spurious=0, hn=0, hn_fp=0)
hel, har = ([], []), ([], [])
for t in tasks:
    r = det.predict(t["img"], imgsz=a.imgsz, conf=a.conf, classes=[0], device="mps", verbose=False)[0]
    W, H = r.orig_shape[1], r.orig_shape[0]
    D = [(x1/W*100, y1/H*100, (x2-x1)/W*100, (y2-y1)/H*100) for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]
    if D:
        F = feats(t["img"], D)
    t["dets"] = D; t["F"] = F if D else None
# fit once, predict all detections (fit_predict trains per call, so batch every test detection together)
allF = np.concatenate([t["F"] for t in tasks if t["F"] is not None])
pred = {k: B.fit_predict(*TR[k], allF, 2) for k in TR}
o = 0
for t in tasks:
    n = len(t["dets"]); pk = {k: pred[k][o:o+n] for k in pred}; o += n
    for g in t["people"]:
        c["pn"] += 1
        m = [j for j, d in enumerate(t["dets"]) if E.iou(g["box"], d) >= 0.5]
        j = max(m, key=lambda j: E.iou(g["box"], t["dets"][j])) if m else None
        c["ptp"] += j is not None
        on = j is not None and pk["on"][j] == 1
        s = g.get("support", "unclear")
        if s in ("stands_on", "climbing"): c["tp" if on else "fn"] += 1
        elif s == "none":
            c["fp"] += on
            if t["split"] == "test_external": c["hn"] += 1; c["hn_fp"] += on
        if j is not None:
            hy = B.HELMET.get(g.get("helmet_state")); zy = {"harness:worn": 1, "harness:not_worn": 0}.get(g.get("harness_state"))
            if hy is not None: hel[0].append(hy); hel[1].append(pk["helmet"][j])
            if zy is not None: har[0].append(zy); har[1].append(pk["harness"][j])
    for j, d in enumerate(t["dets"]):
        if pk["on"][j] == 1 and all(E.iou(g["box"], d) < 0.3 for g in t["people"]): c["spurious"] += 1
fp = c["fp"] + c["spurious"]; P_ = c["tp"]/(c["tp"]+fp) if c["tp"]+fp else 0; R_ = c["tp"]/(c["tp"]+c["fn"])
res = {"name": a.name, "weights": a.weights, "person_recall": round(c["ptp"]/c["pn"], 4), "on_P": round(P_, 4), "on_R": round(R_, 4),
       "on_F1": round(2*P_*R_/(P_+R_), 4) if P_+R_ else 0, "hardneg_fp_rate": round(c["hn_fp"]/c["hn"], 4),
       "helmet_on_found": B.metrics(*hel), "harness_on_found": B.metrics(*har), "counts": c}
json.dump(res, open(OUT / f"{a.name}.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
print(json.dumps({k: res[k] for k in ("name", "person_recall", "on_P", "on_R", "on_F1", "hardneg_fp_rate")}),
      "helmet", res["helmet_on_found"]["bal_acc"], "harness", res["harness_on_found"]["bal_acc"])
