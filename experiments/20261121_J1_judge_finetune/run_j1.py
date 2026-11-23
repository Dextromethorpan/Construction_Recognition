"""J1_judge_finetune driver: j1 crops cache -> per draw fine-tune / MLP -> scores CSVs (judge.py schema) -> threshold, runlog."""
import csv, json, re, subprocess, sys, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
H = Path.home(); C = H / "GenIA-code"; E = Path(__file__).parent; sys.path.insert(0, str(C / "08_controlled")); import judge
PY = str(C / ".venv-train/bin/python"); CFG = f"experiments/{E.name}/config.yaml"; STUDY = "J1_judge_finetune"
cfg = judge.load_cfg(C / "experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg); dets = judge.detections(cfg, imgs)
F, keys = judge.features(cfg, imgs, dets); row = {k: i for i, k in enumerate(keys)}; pc = 0
cache = E / "c1_crops.npy"
if not cache.exists():
    arr = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(len(keys), 224, 224, 3))
    by = {}
    for n, k in enumerate(keys): by.setdefault(k[0], []).append((n, k))
    for g, lst in by.items():
        t = imgs[g]; im = Image.open(t["img"]).convert("RGB"); dp = [d for d in dets.get(g, []) if d["cls"] == pc]
        for n, (gg, kind, i) in lst:
            b = t["people"][i]["box"] if kind == "gt" else tuple(dp[i]["box"]); arr[n] = np.asarray(judge.crop_c1(im, b))
    arr.flush(); print("crops cached", len(keys), flush=True)
CR = np.load(cache, mmap_mode="r")
from ultralytics import YOLO
dev = "mps"; MEAN = torch.zeros(3)

def make_net():
    m = YOLO(str(judge.EMB_W)).model; net = m.model
    head = net[-1]; head.linear = nn.Linear(head.linear.in_features, 2)
    for p in m.parameters(): p.requires_grad = True
    return m.to(dev)

def to_t(idx, aug, rng):
    x = torch.from_numpy(np.stack([CR[i] for i in idx])).float().div(255).permute(0, 3, 1, 2)
    if aug:
        if rng.random() < 1: x = torch.where(torch.rand(len(x), 1, 1, 1) < 0.5, x.flip(3), x)
        x = x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1)) + 0.1 * (torch.rand(len(x), 3, 1, 1) - 0.5)
        s = rng.uniform(0.8, 1.0); sz = int(224 * s); o = rng.randint(0, 224 - sz)
        x = torch.nn.functional.interpolate(x[:, :, o:o + sz, o:o + sz], size=224, mode="bilinear", align_corners=False)
    return x.clamp(0, 1).to(dev)

def finetune(tr_idx, y, seed):
    torch.manual_seed(seed); rng = random.Random(seed); m = make_net(); head = m.model[-1].linear
    hp = set(id(p) for p in head.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in m.parameters() if id(p) not in hp], "lr": 3e-4},
                             {"params": head.parameters(), "lr": 3e-3}], weight_decay=1e-4)
    cnt = np.bincount(y, minlength=2); w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w); EP = 8; steps = EP * ((len(tr_idx) + 63) // 64)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 3e-3], total_steps=steps, pct_start=0.1)
    yt = torch.tensor(y, device=dev)
    for ep in range(EP):
        m.train(); perm = np.random.default_rng(seed * 100 + ep).permutation(len(tr_idx))
        for b in range(0, len(perm), 64):
            bi = perm[b:b + 64]; x = to_t([tr_idx[i] for i in bi], True, rng)
            out = m(x); out = out[0] if isinstance(out, (tuple, list)) else out
            # ultralytics Classify returns softmax in eval and logits in train
            loss = lossf(out, yt[bi]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    def prob(idx):
        ps = []
        with torch.no_grad():
            for b in range(0, len(idx), 128):
                o = m(to_t(idx[b:b + 128], False, None)); o = o[0] if isinstance(o, (tuple, list)) else o
                ps.append(o[:, 1].float().cpu().numpy())   # eval mode: already softmax
        return np.concatenate(ps) if ps else np.zeros(0)
    return prob

def mlp(X, y, seed):
    torch.manual_seed(seed); mu, sd = X.mean(0), X.std(0) + 1e-6; Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    cnt = np.bincount(y, minlength=2); lossf = nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    for ep in range(60):
        net.train(); perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]; l = lossf(net(Xt[i]), yt[i]); opt.zero_grad(); l.backward(); opt.step()
    net.eval()
    return lambda Z: torch.softmax(net(torch.tensor((Z - mu) / sd)), 1)[:, 1].detach().numpy()

sel = sorted(g for g, t in imgs.items() if t["set"] in judge.SEL_SETS); f_on = judge.TARGETS["on"]
cols = ["image_id", "person_id", "set", "on_scaffold", "score", "group_id", "matched", "det_conf", "review_status", "support_surface"]
for arm in sys.argv[1:] or ["J1", "J2"]:
    for d in range(5):
        trg = judge.train_draw(imgs, d, 0.8)
        rr = [(row[(g, "gt", i)], f_on(p)) for g in trg for i, p in enumerate(imgs[g]["people"]) if f_on(p) is not None]
        tr_idx = [r for r, _ in rr]; y = np.array([v for _, v in rr])
        P = finetune(tr_idx, y, d) if arm == "J1" else (lambda m: (lambda idx: m(F["C1"][idx])))(mlp(F["C1"][tr_idx], y, d))
        e2e, cond = [], []
        for g in sel:
            t = imgs[g]; dp = [x for x in dets.get(g, []) if x["cls"] == pc]; gt = [p["box"] for p in t["people"]]
            mt = judge.match(gt, [tuple(x["box"]) for x in dp], 0.5)
            ds = P([row[(g, "det", j)] for j in range(len(dp))]) if dp else np.zeros(0)
            gs = P([row[(g, "gt", i)] for i in range(len(gt))]) if gt else np.zeros(0)
            for i, p in enumerate(t["people"]):
                j = mt.get(i); base = dict(image_id=g, person_id=p["pid"], set=t["set"], on_scaffold=p["on"], group_id=t["group"],
                                           review_status=p["review"], support_surface=p["support"])
                e2e.append({**base, "score": round(float(ds[j]), 6) if j is not None else 0.0, "matched": int(j is not None), "det_conf": dp[j]["conf"] if j is not None else ""})
                cond.append({**base, "score": round(float(gs[i]), 6), "matched": 1, "det_conf": ""})
        for name, rows in ((f"scores_{arm}_d{d}.csv", e2e), (f"scores_cond_{arm}_d{d}.csv", cond)):
            with open(E / name, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        ap = {s: round(judge.ap_score([r["on_scaffold"] == "on" for r in cond if r["set"] == s and r["on_scaffold"] in ("on", "not_on")],
                                      [r["score"] for r in cond if r["set"] == s and r["on_scaffold"] in ("on", "not_on")]), 4) for s in judge.SEL_SETS}
        out = subprocess.run([PY, C / "08_controlled/threshold.py", E / f"scores_{arm}_d{d}.csv"], capture_output=True, text=True).stdout
        (E / f"threshold_{arm}_d{d}.txt").write_text(out)
        m = re.search(r"FA budget 1.5%: threshold (\S+) \| recall (\S+) \((\d+) of (\d+)\) 95% CI \[(\S+), (\S+)\] \| false alarms (\d+) of (\d+)", out)
        m5 = re.search(r"FA budget 5.0%: threshold \S+ \| recall (\S+) \((\d+) of", out)
        rid = subprocess.run([PY, C / "08_controlled/runlog.py", "add", "--study", STUDY, "--arm", arm, "--draw", str(d), "--config", CFG,
                              "--label-version", "v2", "--split-version", "v2", "--series", "real_only", "--status", "ok", "--primary", m.group(2),
                              "--ci", m.group(5), m.group(6), "--threshold", m.group(1), "--n-pos", m.group(4), "--n-neg", m.group(8), "--fa-budget", "0.015",
                              "--notes", f"POST-TEST val only. recall@5% {m5.group(1)} ({m5.group(2)} of {m.group(4)}); cond AP val_v2 {ap['val_v2']} cs10k_dev {ap['cs10k_dev']}"],
                             cwd=C, capture_output=True, text=True).stdout.strip()
        print(arm, d, rid, m.group(2), m5.group(1), ap, flush=True)
