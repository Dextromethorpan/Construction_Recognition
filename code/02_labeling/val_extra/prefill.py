"""Person proposals for val_extra: yolo11m conf 0.15 @1280 UNION D1_960 conf 0.05 @960, merged with NMS 0.5. -> proposals.json"""
import json, torch, torchvision
from pathlib import Path
from ultralytics import YOLO
D = Path.home() / "GenIA-data"; V = D / "labels/relations/v2x"
order = [l.strip() for l in open(V / "val_extra_order.txt") if l.strip()][:400]
ms = [(YOLO(str(D / "runs/yolo11m.pt")), 1280, 0.15, [0]), (YOLO(str(D / "runs/D1_detector/D1_ft960/weights/best.pt")), 960, 0.05, [0])]
out = {}
for g in order:
    p = str(D / "extracted/cs10k" / (g.split("/")[1] + ".jpg")); B, S = [], []
    for m, isz, cf, cls in ms:
        r = m.predict(p, imgsz=isz, conf=cf, classes=cls, device="mps", verbose=False)[0]; B.append(r.boxes.xyxy.cpu()); S.append(r.boxes.conf.cpu()); W, H = r.orig_shape[1], r.orig_shape[0]
    b, s = torch.cat(B), torch.cat(S); k = torchvision.ops.nms(b, s, 0.5); b, s = b[k], s[k]
    out[g] = dict(img=p, W=W, H=H, boxes=[[round(x1 / W * 100, 2), round(y1 / H * 100, 2), round((x2 - x1) / W * 100, 2), round((y2 - y1) / H * 100, 2), round(float(c), 3)]
                                           for (x1, y1, x2, y2), c in zip(b.tolist(), s.tolist())])
json.dump(out, open(V / "proposals.json", "w")); print("images", len(out), "boxes", sum(len(v["boxes"]) for v in out.values()))
