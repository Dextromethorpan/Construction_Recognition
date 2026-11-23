"""ONE-TIME test of the final pipeline on test_v1 (336 real photos, v1 labels). Refuses to run without a decision row
for study `final_pipeline` and refuses a second run (runlog.py check-test). Nothing here may change after it runs.

Pipeline (fixed by RUNLOG r00074 / r00063 / r00095): detector D0 (yolo11n 512, conf 0.25) -> C1 on-scaffold judge,
C0 helmet / harness heads (logistic regression, fit on train_v2 draw d = 80%, seeds 0-4, exactly as in the studies) ->
on-scaffold cutoff per draw = the val FA 1.5% cutoff from D1_detector/scores_D0_d<d>.csv (and 5% as secondary).
Variants scored in the same run: pt_fp32 (MPS) and coreml_int8. Each subset reported separately.
  test_final.py"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image
H = Path.home(); C = H / "GenIA-code"; E = Path(__file__).parent
sys.path.insert(0, str(C / "08_controlled")); sys.path.insert(0, str(C / "07_ablation"))
chk = subprocess.run([sys.executable, C / "08_controlled/runlog.py", "check-test", "--study", "final_pipeline"], capture_output=True, text=True)
if chk.returncode: sys.exit("refused: " + chk.stdout + chk.stderr)
import judge, threshold as T, evaluate as EV
from ultralytics import YOLO
cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
VAR = {"pt_fp32": (str(H / "GenIA-data/runs/ablation_v2b/V2_harness_field/weights/best.pt"), "mps"),
       "coreml_int8": (str(H / "GenIA-data/runs/quantize/D0/coreml_int8.mlpackage"), "cpu")}
tasks = EV.test_tasks(); print("test images", len(tasks), {s: sum(t["split"] == s for t in tasks) for s in ("test_indist", "test_external", "test_heldout")})
ON = {"stands_on": "on", "climbing": "on", "none": "not_on"}
HEL = {"helmet:worn": 1, "helmet:not_worn": 0}; HAR = {"harness:worn": 1, "harness:not_worn": 0}

# --- train-side features (cached from the studies) and the 5 judge / PPE heads
F, keys = judge.features(cfg, imgs, judge.detections(cfg, imgs)); row = {k: i for i, k in enumerate(keys)}
heads = []
for d in range(5):
    trg = judge.train_draw(imgs, d, 0.8); m = {}
    for tname, arm in (("on", "C1"), ("helmet", "C0"), ("harness", "C0")):
        f = judge.TARGETS[tname]; rr = [(row[(g, "gt", i)], f(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f(p) is not None]
        m[tname] = judge.fit(F[arm][[r for r, _ in rr]], np.array([y for _, y in rr]), 1e-3)
    _, _, yv, sv = T.load(C / f"experiments/20261121_D1_detector/scores_D0_d{d}.csv")
    m["cut"] = {b: T.cutoff(yv, sv, b) for b in T.BUDGETS}; heads.append(m)

enc = judge.Encoder(); out = {}
for vname, (w, dev) in VAR.items():
    model = YOLO(w, task="detect"); rows = []
    for t in tasks:
        r = model.predict(t["img"], imgsz=512, conf=0.25, device=dev, verbose=False, max_det=300)[0]
        W, Hh = r.orig_shape[1], r.orig_shape[0]
        dets = [(x1 / W * 100, y1 / Hh * 100, (x2 - x1) / W * 100, (y2 - y1) / Hh * 100)
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if int(c) == 0]
        gt = [p["box"] for p in t["people"]]
        if not gt and not dets: continue
        im = Image.open(t["img"]).convert("RGB"); boxes = gt + dets
        e0, _ = enc([judge.crop_c0(im, b) for b in boxes]); e1, _ = enc([judge.crop_c1(im, b) for b in boxes])
        X1 = np.array([np.concatenate([e1[k], judge.geo6(b, len(gt) if k < len(gt) else len(dets))]) for k, b in enumerate(boxes)], np.float32)
        X0 = np.array([np.concatenate([e0[k], judge.geo6(b, len(gt) if k < len(gt) else len(dets))]) for k, b in enumerate(boxes)], np.float32)
        mt = judge.match(gt, dets, 0.5)
        for i, p in enumerate(t["people"]):
            j = mt.get(i); rec = dict(gid=t["gid"], subset=t["split"], on=ON.get(p.get("support"), "unclear"),
                                      helmet=HEL.get(p.get("helmet_state")), harness=HAR.get(p.get("harness_state")), matched=int(j is not None))
            for d, m in enumerate(heads):
                rec[f"s{d}"] = float(m["on"](X1[[len(gt) + j]])[0]) if j is not None else 0.0
                rec[f"h{d}"] = float(m["helmet"](X0[[i]])[0]); rec[f"z{d}"] = float(m["harness"](X0[[i]])[0])
            rows.append(rec)
        # spurious: detections matching no labeled person, scored by the judge
        for j in range(len(dets)):
            if j not in mt.values() and all(judge.iou(dets[j], b) < 0.3 for b in gt):
                rows.append(dict(gid=t["gid"], subset=t["split"], on="spurious", matched=1,
                                 **{f"s{d}": float(m["on"](X1[[len(gt) + j]])[0]) for d, m in enumerate(heads)}))
    json.dump(rows, open(E / f"rows_{vname}.json", "w"))
    res = {}
    for sub in ("all_real", "test_indist", "test_external", "test_heldout"):
        R = [r for r in rows if sub == "all_real" or r["subset"] == sub]
        pos = [r for r in R if r["on"] == "on"]; neg = [r for r in R if r["on"] == "not_on"]; sp = [r for r in R if r["on"] == "spurious"]
        S = {"n_on": len(pos), "n_not_on": len(neg), "on_boxed": sum(r["matched"] for r in pos), "spurious_dets": len(sp)}
        for b in T.BUDGETS:
            rec_d = [np.mean([r[f"s{d}"] >= heads[d]["cut"][b] for r in pos]) if pos else float("nan") for d in range(5)]
            fa_d = [np.mean([r[f"s{d}"] >= heads[d]["cut"][b] for r in neg]) if neg else float("nan") for d in range(5)]
            sp_d = [sum(r[f"s{d}"] >= heads[d]["cut"][b] for r in sp) for d in range(5)]
            # bootstrap by image for draw-mean recall
            g = np.array([r["gid"] for r in pos]); ug = np.unique(g); rng = np.random.default_rng(0); bs = []
            if len(ug):
                hits = np.array([[r[f"s{d}"] >= heads[d]["cut"][b] for d in range(5)] for r in pos]).mean(1)
                mem = {u: np.where(g == u)[0] for u in ug}
                for _ in range(2000):
                    ix = np.concatenate([mem[u] for u in rng.choice(ug, len(ug))]); bs.append(hits[ix].mean())
            S[f"fa{b}"] = dict(recall=round(float(np.mean(rec_d)), 3), recall_draws=[round(float(x), 3) for x in rec_d],
                               recall_ci=[round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])] if bs else None,
                               hits_mean=round(float(np.mean(rec_d)) * len(pos), 1),
                               false_alarm_rate=round(float(np.mean(fa_d)), 3), spurious_above=round(float(np.mean(sp_d)), 1))
        for k, key in (("helmet", "h"), ("harness", "z")):
            L = [r for r in R if r.get(k) is not None]
            y = np.array([r[k] for r in L]); S[f"{k}_n"] = [int(y.sum()), int(len(y) - y.sum())]
            S[f"{k}_bal_acc"] = round(float(np.mean([judge.bal_acc(y, np.array([r[f"{key}{d}"] >= 0.5 for r in L]).astype(int)) for d in range(5)])), 3) if len(L) else None
        res[sub] = S
    out[vname] = res; print(vname, json.dumps(res["all_real"]), flush=True)
json.dump(out, open(E / "TEST_RESULTS.json", "w"), indent=1)
a = out["pt_fp32"]["all_real"]["fa0.015"]
subprocess.run([sys.executable, C / "08_controlled/runlog.py", "add", "--study", "final_pipeline", "--arm", "D0_C1_C0heads", "--status", "test",
                "--config", "experiments/20261121_D1_detector/config.yaml", "--label-version", "v1", "--split-version", "v2", "--series", "real_only",
                "--primary", str(a["recall"]), "--ci", *map(str, a["recall_ci"]), "--fa-budget", "0.015",
                "--notes", "ONE-TIME test_v1. See experiments/20261123_final_test/TEST_RESULTS.json (pt_fp32 and coreml_int8, per subset)"], check=True)
