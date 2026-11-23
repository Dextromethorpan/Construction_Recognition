#!/bin/zsh
# Step 1 rerun: base trained on the COMPLETED person labels (fix: per-file image symlinks)
set -u
PY=~/GenIA-code/.venv-train/bin/python; A=~/GenIA-code/07_ablation; R=~/GenIA-data/runs
$PY ~/GenIA-code/03_train_yolo/train_detector.py --model yolo11n.pt --imgsz 512 --epochs 15 --name base2_persons_n512_e15 --data ~/GenIA-data/build/B_persons_v1/data.yaml
BASE=$R/detector/base2_persons_n512_e15/weights/best.pt
$PY $A/evaluate.py $BASE base2_only
$PY $A/finetune.py $BASE V2_harness_field ablation_v2 ablation_v2b && \
$PY $A/evaluate.py $R/ablation_v2b/V2_harness_field/weights/best.pt step1b_V2_persons
$PY $A/combined.py $R/ablation_v2b/V2_harness_field/weights/best.pt step2b_own_detector_plus_classifier_neg400 --imgsz 512 --neg 400
echo "=== done $(date)"
