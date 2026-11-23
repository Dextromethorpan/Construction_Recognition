#!/usr/bin/env python3
"""GenIA demo: the FINAL tested pipeline (experiments/20261123_final_test/test_final.py, draw d=0) on any photo(s).

  detector D0 coreml_int8 (yolo11n, imgsz 512, conf 0.25, CPU) -> person boxes
  -> crop C1 (3x2 context) + geo6 -> on-scaffold logistic head, "on" if prob >= val FA-1.5% cutoff
  -> crop C0 (2x1.5 person)  + geo6 -> helmet / harness logistic heads, "worn" if prob >= 0.5
  encoder: frozen yolo11n-cls (judge.Encoder). Heads: 09_demo/heads_d0.npz (made by fit_heads.py).

  .venv-train/bin/python 09_demo/demo.py IMAGE_OR_FOLDER --out DIR
Runs offline: every weight file is local; HF_HUB_OFFLINE / YOLO_OFFLINE are set before ultralytics is imported."""
import argparse, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("YOLO_OFFLINE", "1")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(REPO / "code" / "08_controlled"))
import judge                                                    # crops, geo6, Encoder: identical to the tested pipeline
DET_W = Path(os.environ.get("GENIA_DETECTOR", REPO / "models" / "detector_coreml_int8.mlpackage"))
if (REPO / "models" / "encoder_yolo11n-cls.pt").exists(): judge.EMB_W = REPO / "models" / "encoder_yolo11n-cls.pt"
FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf") if Path(f).exists()), None)
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Pipeline:
    def __init__(self, heads=HERE / "heads_d0.npz"):
        for p in (DET_W, judge.EMB_W, heads):
            if not Path(p).exists(): sys.exit(f"missing local file {p} (demo does not download anything)")
        from ultralytics import YOLO
        z = np.load(heads); self.h = {t: tuple(z[f"{t}_{k}"] for k in ("mu", "sd", "W", "b")) for t in ("on", "helmet", "harness")}
        self.cut = float(z["on_cut"])
        self.det = YOLO(str(DET_W), task="detect"); self.enc = judge.Encoder()
        self.run(Image.new("RGB", (640, 480), (120, 120, 120)))     # warm-up: CoreML compile, MPS init (detector + encoder)
        for k in range(1, 9): self.enc([Image.new("RGB", (224, 224))] * k)

    def prob(self, t, X):
        mu, sd, W, b = self.h[t]; z = ((X - mu) / sd) @ W.T + b; z -= z.max(1, keepdims=True); e = np.exp(z)
        return e[:, 1] / e.sum(1)

    def run(self, im):
        t0 = time.perf_counter()
        r = self.det.predict(im, imgsz=512, conf=0.25, device="cpu", verbose=False, max_det=300)[0]
        t_det = time.perf_counter() - t0
        W, Hh = r.orig_shape[1], r.orig_shape[0]
        dets = [((x1 / W * 100, y1 / Hh * 100, (x2 - x1) / W * 100, (y2 - y1) / Hh * 100), s)
                for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()) if int(c) == 0]
        people = []
        if dets:
            boxes = [b for b, _ in dets]; n = len(boxes)
            e0, _ = self.enc([judge.crop_c0(im, b) for b in boxes]); e1, _ = self.enc([judge.crop_c1(im, b) for b in boxes])
            X0 = np.array([np.concatenate([e0[k], judge.geo6(b, n)]) for k, b in enumerate(boxes)], np.float32)
            X1 = np.array([np.concatenate([e1[k], judge.geo6(b, n)]) for k, b in enumerate(boxes)], np.float32)
            on, hel, har = self.prob("on", X1), self.prob("helmet", X0), self.prob("harness", X0)
            people = [dict(box=b, det_conf=float(s), on=float(on[k]), on_yes=bool(on[k] >= self.cut),
                           helmet=float(hel[k]), harness=float(har[k])) for k, (b, s) in enumerate(dets)]
        return dict(people=people, det_ms=t_det * 1000, total_ms=(time.perf_counter() - t0) * 1000)


def font(sz):
    return ImageFont.truetype(FONT, sz) if FONT else ImageFont.load_default()


def draw(im, res, title=None, width=1400):
    s = width / im.width; im = im.resize((width, round(im.height * s)))
    fs = max(16, width // 70); f, fb = font(fs), font(fs + 4)
    foot = fs * 2 + 16 + (fs + 12 if title else 0)
    out = Image.new("RGB", (im.width, im.height + foot), (20, 20, 20)); out.paste(im, (0, 0)); d = ImageDraw.Draw(out)
    ok, no = ("✓", "✗") if FONT else ("Y", "N"); placed = []
    for k, p in enumerate(sorted(res["people"], key=lambda p: p["box"][1])):
        x, y, w, h = [v / 100 * dim for v, dim in zip(p["box"], (im.width, im.height, im.width, im.height))]
        col = (40, 200, 60) if p["on_yes"] else (170, 170, 170)
        d.rectangle([x, y, x + w, y + h], outline=col, width=max(3, width // 350))
        txt = (f"helmet {ok if p['helmet'] >= .5 else no} {p['helmet']:.2f} · harness {ok if p['harness'] >= .5 else no} {p['harness']:.2f}"
               f" · scaffold {ok if p['on_yes'] else no} {p['on']:.2f}")
        tw, th = d.textbbox((0, 0), txt, font=f)[2:]
        tx = min(max(0, x), im.width - tw - 8); ty = y - th - 10 if y - th - 10 > 0 else min(y + h + 2, im.height - th - 8)
        for _ in range(40):                                     # avoid covering an earlier label: step down until free
            if not any(tx < a2 and tx + tw + 8 > a1 and ty < b2 and ty + th + 8 > b1 for a1, b1, a2, b2 in placed): break
            ty = min(ty + th + 10, im.height - th - 8)
        placed.append((tx, ty, tx + tw + 8, ty + th + 8))
        d.rectangle([tx, ty, tx + tw + 8, ty + th + 8], fill=(0, 0, 0) if not p["on_yes"] else (20, 90, 30))
        d.text((tx + 4, ty + 3), txt, fill=(255, 255, 255), font=f)
    yb = im.height + 8
    if title: d.text((10, yb), title, fill=(255, 255, 255), font=fb); yb += fs + 12
    d.text((10, yb), f"{len(res['people'])} people · detector {res['det_ms']:.0f} ms · total {res['total_ms']:.0f} ms "
                     f"(yolo11n int8 CoreML CPU + yolo11n-cls heads)  ·  green box = on scaffold (cutoff {CUT:.3f})",
           fill=(220, 220, 220), font=f)
    return out


def main():
    global CUT
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None, help="optional caption drawn in the footer (single image)")
    a = ap.parse_args(); src = Path(a.src); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXT) if src.is_dir() else [src]
    t = time.perf_counter(); P = Pipeline(); CUT = P.cut; print(f"models loaded + warm-up {1000 * (time.perf_counter() - t):.0f} ms")
    for fp in files:
        im = Image.open(fp).convert("RGB"); res = P.run(im)
        draw(im, res, a.title).save(out / (fp.stem + "_demo.png"))
        print(f"{fp.name}: {len(res['people'])} people, detector {res['det_ms']:.0f} ms, total {res['total_ms']:.0f} ms | " +
              "; ".join(f"H{p['helmet']:.2f} Z{p['harness']:.2f} S{p['on']:.2f}{'*' if p['on_yes'] else ''}" for p in res["people"]))


CUT = 0.0
if __name__ == "__main__":
    main()
