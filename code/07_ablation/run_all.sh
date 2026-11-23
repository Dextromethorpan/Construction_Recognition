#!/bin/zsh
# Ablation A: wait for the base, score it, then fine-tune + score each variant in turn.
set -u
PY=~/GenIA-code/.venv-train/bin/python; A=~/GenIA-code/07_ablation
BASE=~/GenIA-data/runs/detector/base_n512_e15/weights/best.pt
while pgrep -f "train_detector.py --model yolo11n.pt --imgsz 512" >/dev/null; do sleep 60; done
[ -f $BASE ] || { echo "base missing"; exit 1; }
$PY $A/evaluate.py $BASE base_only
for V in V0_base V1_unclear_class V2_harness_field V3_claude V4_no_synth V5_unclear_neg; do
  echo "=== $V $(date)"
  $PY $A/finetune.py $BASE $V && $PY $A/evaluate.py ~/GenIA-data/runs/ablation/$V/weights/best.pt $V
done
echo "=== done $(date)"
