# The model, and the models tried later

## 1. The final model
It is **two stages**: a small detector finds the people, then small per-person classifiers answer the three questions.

| Stage | Model | Where its knowledge comes from | Size and speed |
|---|---|---|---|
| **1. Detector** | **YOLO11n** (Ultralytics, "nano", about 2.6 M parameters), 512 px input, confidence 0.25 | Starts from COCO pre-training. Trained 15 epochs on 13.6k construction images (scheme B, with the missing person boxes filled in), then fine-tuned 10 epochs with the chosen labeling policy (`V2_harness_field`) | CoreML INT8: **2.86 MB, 4.1 ms** per photo on the M5 |
| **2a. On-scaffold judge** | Logistic regression on a **frozen YOLO11n-cls embedding** (ImageNet-pretrained image classifier, not trained by us) of a **big crop** around the person (3× width, 2× height), plus 6 box numbers (position and size) | Trained on the v2 labels (245 on-scaffold and 1,638 not-on real people in training) | A few milliseconds; the encoder stays FP32 |
| **2b. Helmet head** | Same recipe on the **tight crop** (C0) | v2 labels | Same |
| **2c. Harness head** | Same recipe on the tight crop | v2 labels | Same |
| **Decision** | On-scaffold "yes" if the score is above the cutoff that gives 1.5% false alarms on validation | Fixed before the test | — |

**Why this shape, in one sentence:** the detector is good at "where is a person", and a separate judge looking at the person *with context* is better at "what is this person standing on" than a detector class would be, while staying tiny and explainable.

**Post-test change (validation only):** the helmet head adds a zoomed **head crop** next to the tight crop (+0.051 balanced accuracy on the original validation set, then confirmed on two new sets: +0.133 [+0.017, +0.280] on 120 new photos and +0.077 [+0.028, +0.130] on 221 CS10k people). It has no test number. In the demo, use the tested version and mention this as the next improvement.

---

## 2. Models tried, and what other models would likely do
**Short answer: for small models, very likely yes, within noise.** The evidence says the limit is the data, not the model size.

**What we actually tested**
| Alternative | Result on validation | What it tells us |
|---|---|---|
| **YOLO11s** (about 3.5× bigger detector) | 0.522 vs 0.534 (−0.012) | A bigger model did not find the small people better |
| YOLO11n at **960 px** | 0.573 (+0.039, not significant) | More pixels help more than more parameters |
| Tiling, 1280 px inference | More people boxed, many more junk boxes | Same conclusion |
| **Fine-tuned judge** instead of frozen | 0.515 vs 0.534 | A stronger judge does not help with our amount of labels |
| **MLP judge** instead of logistic | 0.520 | Same |
| Two-stage with **COCO YOLO11m** (a bigger generic detector) as stage 1 (exploration) | More people found, many more false alarms | Bigger generic detector is not the answer either |

**What I would expect from models we did not test** (reasoned, not measured, say it that way):
- **YOLOv8n, YOLOv10n, RT-DETR-small, EfficientDet-D0:** about the same. They are all COCO-pretrained small detectors, and our misses are people that look unlike the training data (crowded, behind tubes, low contrast, other domains). That is a data problem; a different small architecture does not fix it.
- **A larger detector (YOLO11m or l):** probably a few points on small people, at 3 to 10× the cost, and our 11s result suggests the gain is small.
- **A small VLM (Qwen2.5-VL 3B or 7B in 4 bits) end to end:** I would expect it to be **worse at finding people** (RF100-VL: Qwen2.5-VL-72B gets 5.6 mAP zero-shot on unusual classes), much slower (seconds per photo instead of milliseconds), and prone to saying "on scaffold" just because a scaffold is in the photo (What'sUp, Neural Motifs). We did not run it.
- **A VLM as the judge only**, after YOLO, with the person marked in the image (Set-of-Mark): the most interesting untested option. It might reduce the "deck next to scaffold" confusion, at a cost in speed. This is the hybrid plan from Discussion 01.


