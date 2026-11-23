"""List Roboflow versions (image counts, augmentation) for the approved sets. Key read from env."""
import os
from roboflow import Roboflow
SETS = {"rf_harness_main": ("harness-s4xxh", "harness-usugr"), "rf_harness_mks": ("mks-temp", "scaffold-harness"),
        "rf_harness_fypcv": ("fypcv", "safety-harness-iess4"), "rf_scaffold_platform": ("bowens-workspace-3mivc", "scaffold-yolov8-detection"),
        "rf_scaffold_climb": ("chungang-likkc", "scaffold-golkd")}
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
for k, (ws, p) in SETS.items():
    proj = rf.workspace(ws).project(p)
    for v in proj.versions():
        aug = getattr(v, "augmentation", None) or {}
        print(k, v.version.split("/")[-1], getattr(v, "splits", None), "aug:", list(aug.keys()) if isinstance(aug, dict) else aug)
