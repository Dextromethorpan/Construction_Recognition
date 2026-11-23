# Models

| File | What | Size |
|---|---|---|
| `detector_coreml_int8.mlpackage` | Deployment detector: YOLO11n exported to CoreML with INT8 weights (macOS / Apple silicon). This is the model `demo/demo.py` uses and the one reported as "CoreML INT8" in the test. | ~2.9 MB |
| `detector_fp32.pt` | The same detector as PyTorch FP32 weights (Ultralytics checkpoint, run `ablation_v2b/V2_harness_field`). Use it on non-Apple systems. | ~5.5 MB |
| `detector_fp32.onnx` | The same FP32 detector exported to ONNX (opset 18, input 1x3x512x512, made with `yolo export format=onnx imgsz=512 simplify=True`). Runs with onnxruntime on Windows, Linux and macOS; `demo/demo.py` uses it by default off the Mac. | ~10 MB |
| `encoder_yolo11n-cls.pt` | Frozen image encoder used by the per-person heads (stock Ultralytics `yolo11n-cls.pt`, unchanged). The heads themselves are in `demo/heads_d0.npz`. | ~5.8 MB |

**Detector settings:** input 512 px, confidence threshold 0.25. Only **class 0 = `person`** is used downstream; the other classes (helmet, head, harness, scaffold parts, ...) came from training and are ignored by the pipeline. Helmet, harness and on-scaffold answers come from the per-person heads, not from detector classes.

```python
from ultralytics import YOLO
det = YOLO("models/detector_fp32.onnx", task="detect")   # or detector_fp32.pt, or detector_coreml_int8.mlpackage on macOS
r = det.predict("photo.jpg", imgsz=512, conf=0.25, classes=[0])[0]
```

Local training paths stored inside the checkpoint metadata were shortened to `~/...`; weights are unchanged.

**Licence:** trained with and derived from Ultralytics YOLO, so these weights are released under **AGPL-3.0** (see `LICENSE`).
