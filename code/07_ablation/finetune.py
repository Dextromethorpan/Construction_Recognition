"""Fine-tune one variant from the shared base checkpoint. Identical settings and seed for every variant."""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import sys
from ultralytics import YOLO
base, var = sys.argv[1], sys.argv[2]
root = sys.argv[3] if len(sys.argv) > 3 else "ablation_v1"; proj = sys.argv[4] if len(sys.argv) > 4 else "ablation"
YOLO(base).train(data=f"{GENIA_DATA}/build/{root}/{var}/data.yaml",
    epochs=10, imgsz=512, batch=16, device="mps", workers=4, seed=0, deterministic=True,
    optimizer="AdamW", lr0=0.001, warmup_epochs=1, close_mosaic=3, cos_lr=True, plots=False,
    project=GENIA_DATA + "/runs/" + proj, name=var, exist_ok=True)
