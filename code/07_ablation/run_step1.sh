#!/bin/zsh
# Step 1: completed person labels -> new base -> recommended policy (V2: drop unclear + harness field) -> score
set -u
PY=~/GenIA-code/.venv-train/bin/python; A=~/GenIA-code/07_ablation; R=~/GenIA-data/runs
while pgrep -f pseudo_persons.py >/dev/null; do sleep 60; done
ABL_B=~/GenIA-data/build/B_persons_v1 ABL_OUT=~/GenIA-data/build/ablation_v2 ABL_VARIANTS=V2_harness_field $PY $A/build_variants.py
cp ~/GenIA-data/build/ablation_v1/base_data.yaml /dev/null
$PY ~/GenIA-code/03_train_yolo/train_detector.py --model yolo11n.pt --imgsz 512 --epochs 15 --name base2_persons_n512_e15 --data ~/GenIA-data/build/B_persons_v1/data.yaml
BASE=$R/detector/base2_persons_n512_e15/weights/best.pt
$PY $A/evaluate.py $BASE base2_only
$PY $A/finetune.py $BASE V2_harness_field ablation_v2 ablation_v2 && \
$PY $A/evaluate.py $R/ablation_v2/V2_harness_field/weights/best.pt step1_V2_persons
$PY $A/combined.py $R/ablation_v2/V2_harness_field/weights/best.pt step2_own_detector_plus_classifier --imgsz 512
echo "=== done $(date)"
