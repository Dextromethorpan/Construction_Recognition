#!/usr/bin/env python3
"""Phase 2 - download approved datasets into ~/GenIA-data/raw/<id>/ exactly as published.

Idempotent: a dataset whose raw/<id>/.done marker exists is skipped.
Keys are read from the environment (HF_TOKEN, ROBOFLOW_API_KEY), never printed.

Usage:
  python 02_download.py <dataset_id> [...]
  python 02_download.py --register <dataset_id> <download_url>   # files placed by hand in raw/<id>/
"""
import datetime
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path.home() / "GenIA-data"
RAW = ROOT / "raw"
MANIFEST = ROOT / "manifest.yaml"
LOG = ROOT / "reports" / "download_log.txt"
MAX_GB_WITHOUT_ASKING = 5.0
TOTAL_BUDGET_GB = 40.0

ROBOFLOW = {
    "hardhat_wkrs": ("joseph-nelson", "hard-hat-workers", 2),
    "rf_harness_main": ("harness-s4xxh", "harness-usugr", 2),
    "rf_harness_mks": ("mks-temp", "scaffold-harness", None),
    "rf_harness_fypcv": ("fypcv", "safety-harness-iess4", 1),
    "rf_scaffold_platform": ("bowens-workspace-3mivc", "scaffold-yolov8-detection", None),
    "rf_scaffold_climb": ("chungang-likkc", "scaffold-golkd", None),
    # added 2026-11-18 after SODA was dropped (raw, unaugmented versions)
    "rf_scaffold_esco": ("esco-7veqt", "scaffold-7bi3u", 2),
    "rf_scaffold_archikhu": ("archikhu", "scaffold-0726-ddew4", 1),
    "rf_scaffold_sparks": ("sparks-f4jed", "scaffold-a4l14", 2),
    "rf_scaffold_mobile": ("chungang-likkc", "mobile-scaffold", 7),
    "rf_scaffold_hook": ("objectdetectionproject-vp2dx", "worker-scaffold-hook", 2),
}


def log(msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_gb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9


def check_budget():
    used = dir_gb(RAW) if RAW.exists() else 0.0
    if used >= TOTAL_BUDGET_GB:
        sys.exit(f"STOP: raw/ already uses {used:.1f} GB (budget {TOTAL_BUDGET_GB} GB)")
    return used


def record(ds_id, download_url, version=None):
    out = RAW / ds_id
    files = sorted(f for f in out.rglob("*") if f.is_file() and f.name != ".done")
    # hash archives and top-level files; for exploded exports hash every file into one digest list
    sums = {str(f.relative_to(out)): sha256(f) for f in files}
    (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    m = yaml.safe_load(MANIFEST.read_text())
    for d in m["datasets"]:
        if d["id"] == ds_id:
            d["status"] = "downloaded"
            d["download_url"] = download_url
            d["downloaded_at"] = datetime.date.today().isoformat()
            d["sha256"] = {"SHA256SUMS": sha256(out / "SHA256SUMS"), "n_files": len(sums)}
            d["download_gb"] = round(dir_gb(out), 3)
            if version is not None:
                d["version"] = version
    MANIFEST.write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True, width=200))
    (out / ".done").write_text(datetime.datetime.now().isoformat())
    log(f"{ds_id}: recorded {len(sums)} files, {dir_gb(out):.2f} GB")


def dl_shwd(out):
    import gdown
    fid = "1qWm7rrwvjAWs1slymbrLaCf7Q-wnGLEX"
    gdown.download(id=fid, output=str(out) + "/", quiet=False)
    return f"https://drive.google.com/uc?id={fid}", "VOC2028"


def dl_cs10k(out):
    from huggingface_hub import snapshot_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("cs10k needs HF_TOKEN (after accepting the dataset terms)")
    snapshot_download("LouisChen15/ConstructionSite", repo_type="dataset", local_dir=out, token=tok)
    return "hf://datasets/LouisChen15/ConstructionSite", "main"


def dl_roboflow(ds_id, out):
    from roboflow import Roboflow
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit(f"{ds_id} needs ROBOFLOW_API_KEY")
    ws, proj, ver = ROBOFLOW[ds_id]
    p = Roboflow(api_key=key).workspace(ws).project(proj)
    if ver is None:
        ver = max(int(v.version.split("/")[-1]) for v in p.versions())
    p.version(ver).download("coco", location=str(out), overwrite=True)
    return f"https://universe.roboflow.com/{ws}/{proj}/dataset/{ver}", f"v{ver}"


def main(argv):
    if argv[:1] == ["--register"]:
        record(argv[1], argv[2])
        return
    for ds_id in argv:
        out = RAW / ds_id
        if (out / ".done").exists():
            log(f"{ds_id}: already downloaded, skipping")
            continue
        used = check_budget()
        log(f"{ds_id}: starting (raw/ uses {used:.2f} GB)")
        out.mkdir(parents=True, exist_ok=True)
        if ds_id == "shwd":
            url, ver = dl_shwd(out)
        elif ds_id == "cs10k":
            url, ver = dl_cs10k(out)
        elif ds_id in ROBOFLOW:
            url, ver = dl_roboflow(ds_id, out)
        else:
            sys.exit(f"{ds_id}: no automatic downloader (place files by hand, then --register)")
        record(ds_id, url, ver)


if __name__ == "__main__":
    main(sys.argv[1:])
