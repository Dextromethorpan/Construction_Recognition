#!/usr/bin/env python3
"""NOT RUN YET (needs Luciano's OK: it runs a model and downloads yolo11n-pose.pt if missing).

Adds COCO-pose suggestions to the v2 import file, as editable predictions:
  - ankle keypoints (left/right) for every person box: keypoint conf >= KP_VIS -> visible label,
    KP_OCC <= conf < KP_VIS -> *_occluded label, else not placed;
  - person boxes the pose model finds that match no existing person box (IoU < 0.5), conf >= BOX_CONF.
    Needed for cs10k_dev (CS10k has no person boxes) and to raise person coverage elsewhere.
Reads  labeling_queue/relation_v2/label_studio_import.json
Writes labeling_queue/relation_v2/label_studio_import_prefilled.json (the original is kept).
Run in the training venv on the Mac: ~/GenIA-code/.venv-train/bin/python prefill_pose.py [--device mps]
"""
import argparse, json, os
from pathlib import Path

D = Path(os.environ.get("GENIA_DATA", Path.home() / "GenIA-data"))
Q = D / "labeling_queue/relation_v2"
BOX_CONF, KP_VIS, KP_OCC = 0.5, 0.5, 0.25
L_ANKLE, R_ANKLE = 15, 16   # COCO keypoint indices


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--weights", default="yolo11n-pose.pt")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    tasks = json.load(open(Q / "label_studio_import.json"))
    cache = {}
    n_box = n_kp = 0
    for t in tasks:
        d = t["data"]
        img = D / d["image"].split("?d=")[1]
        if d["global_id"] not in cache:
            r = model.predict(str(img), device=a.device, verbose=False)[0]
            H, W = r.orig_shape
            dets = []
            for b, c, k in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist(), r.keypoints.data.tolist()):
                box = (100 * (b[0] - b[2] / 2), 100 * (b[1] - b[3] / 2), 100 * b[2], 100 * b[3])
                dets.append((box, c, [(100 * k[i][0] / W, 100 * k[i][1] / H, k[i][2]) for i in (L_ANKLE, R_ANKLE)]))
            cache[d["global_id"]] = (dets, W, H)
        dets, W, H = cache[d["global_id"]]
        pred = t["predictions"][0] if t["predictions"] else {"model_version": "", "result": []}
        res = pred["result"]
        persons = [(r["id"], (r["value"]["x"], r["value"]["y"], r["value"]["width"], r["value"]["height"]))
                   for r in res if r.get("from_name") == "box" and r["value"]["rectanglelabels"][0] == "person"]
        has_kp = any(r.get("type") == "keypointlabels" for r in res)
        kp_done = set()   # a person box gets ankles from at most one pose detection (highest conf first)
        for j, (box, c, ank) in enumerate(dets):
            match = max(persons, key=lambda p: iou(p[1], box), default=None)
            if match is None or iou(match[1], box) < 0.5:
                if c < BOX_CONF: continue
                pid = f"pose{j}"
                res.append({"id": pid, "from_name": "box", "to_name": "image", "type": "rectanglelabels",
                            "original_width": W, "original_height": H,
                            "value": {"x": box[0], "y": box[1], "width": box[2], "height": box[3], "rotation": 0,
                                      "rectanglelabels": ["person"]}})
                n_box += 1
            else:
                pid = match[0]
            if has_kp and d["pass"] == 1:
                continue   # keep existing (hand-checked) ankles
            if d["pass"] == 2:
                continue   # pass 2 stays geometry-only for boxes; no keypoint suggestions
            if pid in kp_done: continue
            kp_done.add(pid)
            for side, (x, y, v) in zip(("left_ankle", "right_ankle"), ank):
                if v < KP_OCC: continue
                lab = side if v >= KP_VIS else side + "_occluded"
                res.append({"id": f"{pid}_{side}", "from_name": "kp", "to_name": "image", "type": "keypointlabels",
                            "original_width": W, "original_height": H,
                            "value": {"x": x, "y": y, "width": 0.6, "keypointlabels": [lab]}})
                n_kp += 1
        if res and not t["predictions"]:
            t["predictions"] = [pred]
        if t["predictions"]:
            t["predictions"][0]["model_version"] = "+".join(filter(None, [t["predictions"][0].get("model_version"), "pose_prefill"]))
    (Q / "label_studio_import_prefilled.json").write_text(json.dumps(tasks))
    print(f"added {n_box} person boxes and {n_kp} ankle points over {len(cache)} images")


if __name__ == "__main__":
    main()
