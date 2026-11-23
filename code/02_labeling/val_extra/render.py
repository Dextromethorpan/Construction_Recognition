"""render.py START N -> sheet images of val_extra tasks START..START+N-1 with a 10% grid and numbered proposal boxes."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path.home() / "GenIA-data"; V = D / "labels/relations/v2x"; OUT = Path(__file__).parent / "_render"; OUT.mkdir(exist_ok=True)
P = json.load(open(V / "proposals.json")); order = [l.strip() for l in open(V / "val_extra_order.txt") if l.strip()][:400]
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
a, n = int(sys.argv[1]), int(sys.argv[2]); crop = [float(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
for k in range(a, a + n):
    g = order[k]; t = P[g]; im = Image.open(t["img"]).convert("RGB"); W, H = im.size
    x0, y0, x1, y1 = crop or (0, 0, 100, 100); im = im.crop((W * x0 / 100, H * y0 / 100, W * x1 / 100, H * y1 / 100))
    s = 1100 / max(im.size); im = im.resize((int(im.width * s), int(im.height * s))); d = ImageDraw.Draw(im); cw, ch = im.size
    px = lambda x, y: ((x - x0) / (x1 - x0) * cw, (y - y0) / (y1 - y0) * ch)
    step = 5 if crop else 10
    for v in range(0, 101, step):
        if x0 <= v <= x1: X, _ = px(v, 0); d.line([(X, 0), (X, ch)], fill=(0, 255, 255), width=1); d.text((X + 2, 2), str(v), fill=(0, 255, 255), font=font)
        if y0 <= v <= y1: _, Y = px(0, v); d.line([(0, Y), (cw, Y)], fill=(0, 255, 255), width=1); d.text((2, Y + 2), str(v), fill=(0, 255, 255), font=font)
    for i, (x, y, w, h, c) in enumerate(t["boxes"]):
        (X1, Y1), (X2, Y2) = px(x, y), px(x + w, y + h); col = (255, 0, 0) if c >= 0.3 else (255, 140, 0)
        d.rectangle([X1, Y1, X2, Y2], outline=col, width=2); d.text(((X1 + X2) / 2 - 5, (Y1 + Y2) / 2 - 8), str(i), fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((cw - 160, ch - 20), f"#{k} {g}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(OUT / f"{k:03d}.jpg", quality=85)
print("ok")
