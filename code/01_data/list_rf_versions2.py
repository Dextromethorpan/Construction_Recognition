"""List versions of the added scaffold sets (key from env)."""
import os
from roboflow import Roboflow
SETS = {"rf_scaffold_esco": ("esco-7veqt", "scaffold-7bi3u"), "rf_scaffold_archikhu": ("archikhu", "scaffold-0726-ddew4"),
        "rf_scaffold_sparks": ("sparks-f4jed", "scaffold-a4l14"), "rf_scaffold_mobile": ("chungang-likkc", "mobile-scaffold"),
        "rf_scaffold_hook": ("objectdetectionproject-vp2dx", "worker-scaffold-hook")}
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
for k, (ws, p) in SETS.items():
    for v in rf.workspace(ws).project(p).versions():
        aug = getattr(v, "augmentation", None) or {}
        print(k, v.version.split("/")[-1], getattr(v, "splits", None), "aug:", list(aug.keys()) if isinstance(aug, dict) else aug)
