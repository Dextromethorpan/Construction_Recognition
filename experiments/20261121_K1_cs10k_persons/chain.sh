#!/bin/zsh
# waits for build_k1.py, then fine-tunes K1 at 960 and scores it (judge C1, 5 draws) vs D0
set -u
PY=~/GenIA-code/.venv-train/bin/python; R=~/GenIA-data/runs; E=~/GenIA-code/experiments/20261121_K1_cs10k_persons
while pgrep -f build_k1.py >/dev/null; do sleep 30; done
test -f ~/GenIA-data/build/K1_cs10k_persons/data.yaml || { echo "build failed"; exit 1; }
echo "=== K1 FT start $(date)"
cd $R && $PY - <<PYEOF
from ultralytics import YOLO
YOLO("$R/detector/base2_persons_n512_e15/weights/best.pt").train(data="${GENIA_DATA:-$HOME/GenIA-data}/build/K1_cs10k_persons/data.yaml",
    epochs=10, imgsz=960, batch=8, device="mps", workers=4, seed=0, deterministic=True,
    optimizer="AdamW", lr0=0.001, warmup_epochs=1, close_mosaic=3, cos_lr=True, plots=False,
    project="$R/K1_cs10k_persons", name="K1_ft960", exist_ok=True)
PYEOF
echo "=== K1 FT done $(date)"
cd ~/GenIA-code
for d in 0 1 2 3 4; do for p in scores scores_cond; do cp experiments/20261121_D1_detector/${p}_D0_d$d.csv $E/${p}_D0_d$d.csv; cp experiments/20261121_D1_detector/${p}_D1_960_d$d.csv $E/${p}_D1_960_d$d.csv; done; done
$PY experiments/20261121_D1_detector/run_arm.py K1 $R/K1_cs10k_persons/K1_ft960/weights/best.pt 960 --study K1_cs10k_persons --config experiments/20261121_K1_cs10k_persons/config.yaml
cd $E && $PY ~/GenIA-code/08_controlled/compare.py . --arm K1 --baseline D0 && $PY ~/GenIA-code/08_controlled/compare.py . --arm K1 --baseline D1_960
echo "=== all done $(date)"
