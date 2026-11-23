"""Short speed benchmark on the M5 (MPS): yolo11n, 1 epoch on 25% of B_objects_platform_v1 at two image sizes."""
import os; GENIA_DATA = os.environ.get("GENIA_DATA", os.path.expanduser("~/GenIA-data"))
import time, json, sys
from ultralytics import YOLO
DATA = GENIA_DATA + "/build/B_objects_platform_v1/data.yaml"
out = {}
for imgsz in (512, 640):
    t = time.time()
    YOLO("yolo11n.pt").train(data=DATA, epochs=1, imgsz=imgsz, batch=16, device="mps", workers=4,
                             fraction=0.25, val=False, plots=False, seed=0, deterministic=False,
                             project=GENIA_DATA + "/runs/bench", name=f"n{imgsz}", exist_ok=True)
    s = time.time() - t
    out[imgsz] = {"sec_quarter_epoch": round(s), "est_min_full_epoch": round(4 * s / 60, 1)}
    print(out, flush=True)
json.dump(out, open(GENIA_DATA + "/runs/bench/result.json", "w"), indent=1)
