# Quantization


## What quantization means (in one paragraph)
A trained network is a long list of numbers (weights). Normally each one is stored as a 32-bit float (FP32). Quantization stores them with fewer bits: **FP16** (16-bit floats) or **INT8** (8-bit integers, 256 levels per channel with a scale factor). Fewer bits means a smaller file and, on hardware that supports it, faster inference. The risk is accuracy: rounding errors can make the model miss small or faint objects.

There are two kinds of INT8:
- **Weight-only INT8:** only the stored weights are 8-bit; they are expanded back when the model runs. The calculations themselves stay in floating point. Small file, very low risk.
- **Full (static) INT8:** weights **and** the intermediate values (activations) are 8-bit. The ranges of the activations must be measured first on **calibration images**. Faster on CPUs, but riskier.

## What we quantized
**Only the detector** (yolo11n, 512 px, the D0 model). It is the part that runs on every photo. The judge's encoder and the three small heads stayed in FP32 and were not changed, so any difference comes from the detector alone.

## The rule, written before any run (21 Nov, ~17:20)
- Recommend the **smallest** variant whose end-to-end on-scaffold recall drops by **at most 0.02** compared with the original, **and** which boxes at most 2 fewer on-scaffold people (of 118).
- Calibration uses **200 training images only**, never validation or test.
- Hypothesis written in advance: FP16 is free; CoreML weight INT8 costs under 0.02; full INT8 costs more on small people, but is about 2× faster on CPU.

## How it was implemented (`06_quantize/quantize.py`)
| Variant | How it was made | Runs on |
|---|---|---|
| PyTorch FP32 (reference) | The trained `.pt` file, unchanged | Mac GPU (MPS) |
| CoreML FP16 | `YOLO(w).export(format="coreml", half=True)` | CoreML (Apple's runtime: Neural Engine, GPU or CPU) |
| **CoreML INT8** | `YOLO(w).export(format="coreml", int8=True)`: **weight-only** 8-bit (linear quantization of the weights by coremltools) | CoreML |
| ONNX FP32 | `YOLO(w).export(format="onnx", simplify=True)` | CPU (onnxruntime) |
| ONNX INT8 | onnxruntime `quantize_static`: **full INT8** in QDQ format, weights per channel, activations calibrated on the 200 training images (letterboxed exactly like YOLO's own preprocessing). The last decoding steps of the detection head (concat, sigmoid, box arithmetic) were **kept in float**, because they turn raw outputs into coordinates and break easily | CPU |

NMS was left out of the exported graphs (`nms=False`) and done by Ultralytics afterwards, so every variant shares the same post-processing.

## How it was measured
- **Accuracy:** each variant re-detected all validation images (`detect_variant.py`). The detections went through **the same judge, the same 5 training draws and the same rule for the 1.5% false-alarm cutoff** as every other study. The measure is end-to-end on-scaffold recall, so it shows what quantization does to the final answer, not just to box scores.
- **Speed:** median over 100 validation photos, one photo at a time, after 10 warm-up runs, **including** preprocessing and NMS, on the M5.
- **Size:** megabytes on disk.

## Results (validation)
| Variant | Size | Speed (median) | Recall at 1.5% FA | People boxed (of 118) | Drop |
|---|---|---|---|---|---|
| PyTorch FP32 | 5.49 MB | 9.3 ms | 0.534 | 83 | — |
| CoreML FP16 | 5.40 MB | 4.2 ms | 0.532 | 83 | 0.002 |
| **CoreML INT8** | **2.86 MB** | **4.1 ms** | **0.534** | **83** | **0.000 → chosen** |
| ONNX FP32 | 10.61 MB | 22.8 ms | 0.529 | 82 | 0.005 |
| ONNX INT8 | 3.19 MB | 18.7 ms | 0.508 | 82 | 0.026 (over budget) |

**Confirmed on the one-time test:** PyTorch FP32 0.517, CoreML INT8 **0.521**, false alarms 5.4% vs 5.2%, helmet and harness unchanged (0.734 and 0.779). The difference is noise.

## How the results changed, and why (the honest reading)
1. **Weight-only INT8 was free.** Rounding the weights to 8 bits per channel barely moves them, so the boxes and scores are almost identical. Size halved: 5.49 → 2.86 MB.
2. **The speed gain comes from CoreML, not from INT8.** FP16 and INT8 run at the same speed (4.2 vs 4.1 ms), because weight-only INT8 still computes in floating point. Moving from PyTorch to Apple's runtime is what halved the time (9.3 → 4.1 ms). Say this plainly; it shows you understand what you measured.
3. **Full INT8 cost accuracy, as predicted, but for less speed than predicted.** Quantizing the activations lost 0.026 recall (over the 0.02 budget), and it was only about 1.2× faster than ONNX FP32 on CPU, not the hoped-for 2×. The hypothesis was half right, and the rule rejected it.
4. **The `.pt` file is already small** (5.49 MB) because Ultralytics stores weights in FP16, which is why "FP32 → FP16" saves almost nothing on disk.
5. **What was not quantized:** the judge encoder (yolo11n-cls). Quantizing it is the natural next step if the whole pipeline must shrink.

## What it means for GenIA
For cloud inference at GenIA, the same method applies with a different target: export to ONNX or TensorRT, check the accuracy with the same end-to-end rule, and pick the smallest variant inside a budget agreed in advance. On edge devices (robots, cameras), weight-only INT8 is the safe first step, and full INT8 needs a calibration set from the real cameras.

