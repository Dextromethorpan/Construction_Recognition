"""Study 0: contact sheets of 30 people from the old combined.py --neg 400 pool and 30 rf_harness_fypcv replay people
(fine-tune replay of the fixed detector). Purpose: are any of these 'not on scaffold' people actually at height?"""
import random, csv
from pathlib import Path
from PIL import Image, ImageDraw
D = Path.home() / "GenIA-data"
def boxes(lbl):
    out = []
    for l in open(lbl):
        if l.startswith("0 "):
            cx, cy, w, h = map(float, l.split()[1:5]); out.append(((cx - w/2)*100, (cy - h/2)*100, w*100, h*100))
    return out
# 1) neg400, same selection as 07_ablation/combined.py
PB = D / "build/B_persons_v1/train"
stems = sorted(p.stem for p in (PB/"labels").iterdir() if p.stem.split("__")[0] in ("hardhat_wkrs", "shwd"))
random.Random(0).shuffle(stems); neg = []
for st in stems:
    bx = [b for b in boxes(PB/"labels"/f"{st}.txt") if b[2]*b[3]/1e4 > 0.002][:3]
    neg += [(str(PB/"images"/f"{st}.jpg"), b) for b in bx]
    if len(neg) >= 400: break
neg = neg[:400]
# 2) fypcv replay people in the V2 fine-tune build
FB = D / "build/ablation_v2/V2_harness_field/train"
fy = [(str(FB/"images"/(p.stem + ext)), b) for p in sorted((FB/"labels").glob("rf_harness_fypcv__*.txt")) for b in boxes(p)
      for ext in [next((e for e in (".jpg", ".jpeg", ".png") if (FB/"images"/(p.stem + e)).exists()), ".jpg")]]
def sheet(items, out, title):
    T = 260; cols = 6; rows = (len(items) + cols - 1)//cols
    S = Image.new("RGB", (cols*T, rows*T + 20), "white"); d = ImageDraw.Draw(S); d.text((4, 4), title, fill="black")
    meta = []
    for k, (img, (x, y, w, h)) in enumerate(items):
        im = Image.open(img).convert("RGB"); W, H = im.size
        cx, cy = (x + w/2)/100*W, (y + h/2)/100*H; s = max(w/100*W, h/100*H) * 1.6
        c = im.crop((cx - s, cy - s, cx + s, cy + s)); sc = T / c.size[0]; c = c.resize((T, T))
        dd = ImageDraw.Draw(c); bx = ((x/100*W - (cx - s))*sc, (y/100*H - (cy - s))*sc)
        dd.rectangle([bx[0], bx[1], bx[0] + w/100*W*sc, bx[1] + h/100*H*sc], outline="red", width=2); dd.text((3, 3), str(k), fill="yellow")
        S.paste(c, ((k % cols)*T, (k//cols)*T + 20)); meta.append((k, img, round(x, 1), round(y, 1), round(w, 1), round(h, 1)))
    S.save(out); csv.writer(open(out.replace(".jpg", ".csv"), "w")).writerows([("k", "image", "x", "y", "w", "h")] + meta)
rng = random.Random(20261121)
print("neg400 pool", len(neg), "fypcv replay people", len(fy), "fypcv images", len({i for i, _ in fy}))
sheet(rng.sample(neg, 30), "sheet_neg400.jpg", "30 of the combined.py --neg 400 people (hardhat_wkrs / shwd)")
sheet(rng.sample(fy, min(30, len(fy))), "sheet_fypcv_replay.jpg", "30 rf_harness_fypcv replay people (V2 fine-tune train)")
