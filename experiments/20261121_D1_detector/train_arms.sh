#!/bin/zsh
# D1_detector training chain: D1 (FT at 1280) then D2 (yolo11s base + FT at 512). Logs to train_arms.log
set -u
PY=~/GenIA-code/.venv-train/bin/python; R=~/GenIA-data/runs
BASE=$R/detector/base2_persons_n512_e15/weights/best.pt
echo "=== D1_960 start $(date)"
$PY - <<PYEOF
from ultralytics import YOLO
YOLO("$BASE").train(data="${GENIA_DATA:-$HOME/GenIA-data}/build/ablation_v2/V2_harness_field/data.yaml",
    epochs=10, imgsz=960, batch=8, device="mps", workers=4, seed=0, deterministic=True,
    optimizer="AdamW", lr0=0.001, warmup_epochs=1, close_mosaic=3, cos_lr=True, plots=False,
    project="$R/D1_detector", name="D1_ft960", exist_ok=True)
PYEOF
echo "=== D1_960 done $(date)"
echo "=== D2 base start $(date)"
$PY ~/GenIA-code/03_train_yolo/train_detector.py --model yolo11s.pt --imgsz 512 --epochs 15 --name base2_persons_s512_e15 --data ~/GenIA-data/build/B_persons_v1/data.yaml
echo "=== D2 FT start $(date)"
$PY ~/GenIA-code/07_ablation/finetune.py $R/detector/base2_persons_s512_e15/weights/best.pt V2_harness_field ablation_v2 D1_detector_D2
echo "=== all done $(date)"
