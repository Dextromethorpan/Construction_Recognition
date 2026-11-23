"""Stage 3: export one detector to FP16 / INT8 variants, measure size and latency.
  quantize.py WEIGHTS IMGSZ OUTDIR
Variants: pt_fp32 (MPS), coreml_fp16, coreml_int8 (weights 8-bit), onnx_fp32 (CPU), onnx_int8 (onnxruntime static QDQ,
calibrated on 200 train_v2 images, never val/test). Latency: 100 val_v2 images, batch 1, 10 warm-up, median ms of
model.predict (pre + inference + NMS). Writes OUTDIR/variants.json; accuracy is scored separately with run_arm.py."""
import json, shutil, sys, time, random
from pathlib import Path
import numpy as np
H = Path.home(); sys.path.insert(0, str(H / "GenIA-code/08_controlled")); import judge
from ultralytics import YOLO
W, IMGSZ, OUT = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
cfg = judge.load_cfg(H / "GenIA-code/experiments/20261121_D1_detector/arm_configs/D0.yaml"); imgs = judge.load_people(cfg)
val = sorted(g for g, t in imgs.items() if t["set"] == "val_v2"); random.Random(0).shuffle(val); val = [imgs[g]["img"] for g in val[:100]]
tr = [l.strip() for l in open(H / "GenIA-data/splits/v2/train_v2.txt") if l.strip()]
def size(p):
    p = Path(p); return round((sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size) / 1e6, 2)
def ensure(src, name):
    dst = OUT / name
    if dst.exists(): shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(src), str(dst)); return str(dst)

V = {}
pt = OUT / "pt_fp32.pt"; shutil.copy(W, pt); V["pt_fp32"] = dict(path=str(pt), device="mps")
V["coreml_fp16"] = dict(path=ensure(YOLO(W).export(format="coreml", imgsz=IMGSZ, half=True, nms=False), "coreml_fp16.mlpackage"), device="cpu")
V["coreml_int8"] = dict(path=ensure(YOLO(W).export(format="coreml", imgsz=IMGSZ, int8=True, nms=False), "coreml_int8.mlpackage"), device="cpu")
V["onnx_fp32"] = dict(path=ensure(YOLO(W).export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=False), "onnx_fp32.onnx"), device="cpu")

# ONNX INT8: static QDQ, calibration = 200 train_v2 images letterboxed exactly like ultralytics preprocessing
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, quant_pre_process
from ultralytics.data.augment import LetterBox
import cv2, onnxruntime as ort
inp = ort.InferenceSession(V["onnx_fp32"]["path"]).get_inputs()[0].name
calib_paths = []
for line in tr:
    p = line.split(",")[0]
    pp = Path(p) if Path(p).is_absolute() else H / "GenIA-data" / p
    if pp.suffix.lower() in (".jpg", ".jpeg", ".png") and pp.exists(): calib_paths.append(str(pp))
if not calib_paths:   # train_v2.txt holds ids: fall back to judge's image table
    calib_paths = [imgs[g]["img"] for g in sorted(imgs) if imgs[g]["set"] == "train_v2"]
random.Random(1).shuffle(calib_paths); calib_paths = calib_paths[:200]
class R(CalibrationDataReader):
    def __init__(s): s.it = iter(calib_paths); s.lb = LetterBox((IMGSZ, IMGSZ), auto=False)
    def get_next(s):
        p = next(s.it, None)
        if p is None: return None
        im = s.lb(image=cv2.imread(p))[..., ::-1].transpose(2, 0, 1)
        return {inp: (np.ascontiguousarray(im)[None].astype(np.float32) / 255.)}
pre = OUT / "onnx_pre.onnx"; quant_pre_process(V["onnx_fp32"]["path"], str(pre))
q = OUT / "onnx_int8.onnx"
quantize_static(str(pre), str(q), R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                per_channel=True, nodes_to_exclude=[n.name for n in __import__("onnx").load(str(pre)).graph.node if "/model.23/" in n.name and n.op_type in ("Concat", "Split", "Sigmoid", "Mul", "Add", "Sub", "Div", "Softmax")])
pre.unlink(); V["onnx_int8"] = dict(path=str(q), device="cpu")
for k, v in V.items():
    m = YOLO(v["path"], task="detect")
    for p in val[:10]: m.predict(p, imgsz=IMGSZ, conf=0.25, device=v["device"], verbose=False)
    ts = []
    for p in val:
        t = time.perf_counter(); m.predict(p, imgsz=IMGSZ, conf=0.25, device=v["device"], verbose=False); ts.append((time.perf_counter() - t) * 1000)
    v.update(size_mb=size(v["path"]), ms_median=round(float(np.median(ts)), 1), ms_p90=round(float(np.percentile(ts, 90)), 1), imgsz=IMGSZ)
    print(k, v, flush=True)
json.dump(V, open(OUT / "variants.json", "w"), indent=1)
