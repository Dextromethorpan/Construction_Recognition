"""Read a Label Studio export (relation v1 OR v2) into one per-person CSV.

Rules:
- A field that was never set is 'unannotated'. It is NEVER turned into 'no' / 'not_on'.
- v1 rows get on_scaffold from v1 'support'; v2 rows from 'support_surface' (C_relation_v2.yaml).
- support_v1 is filled for both (v2 via the backward map), so v1 and v2 can be scored together.
- Consistency problems are written to a 'flags' column for review, never fixed here.

Usage: parse_v2.py EXPORT.json OUT.csv [--annotation last|first] [--frozen-v1 codex_reviewed.json]
       --frozen-v1 adds 'v1_support' for relabel_v1 tasks (matched by global_id + IoU >= 0.5) and flags disagreements.
"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from v2_fields import (UNANNOTATED, PERSON_FIELDS, ON_SCAFFOLD, TO_V1_SUPPORT, V1_ON, KEYPOINTS,
                       strip, hard_negative_type, consistency_flags)

V1_FIELDS = ["support", "proximity", "helmet_state", "harness_state", "tied_off", "feet_on", "visibility"]


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def people_of(result):
    """Return (people dict by region id, image-level choices dict)."""
    people, image = {}, {}
    kps = []
    for r in result:
        v, t, fn = r.get("value", {}), r.get("type"), r.get("from_name")
        if t == "rectanglelabels" and fn == "box" and v.get("rectanglelabels", [""])[0] == "person":
            people.setdefault(r["id"], {})["box"] = (v["x"], v["y"], v["width"], v["height"])
        elif t == "choices":
            ch = v.get("choices", [])
            if r.get("id") and fn not in ("all_people_boxed", "viewpoint", "distance", "lighting", "occlusion", "image_flag"):
                people.setdefault(r["id"], {})[fn] = ch[0] if len(ch) == 1 else "|".join(ch)
            else:
                image[fn] = ch[0] if ch else UNANNOTATED
        elif t == "keypointlabels":
            kps.append((v["keypointlabels"][0], v["x"], v["y"]))
    people = {k: p for k, p in people.items() if "box" in p}
    # assign each ankle keypoint to the person box that contains it (smallest box wins)
    for lab, x, y in kps:
        if lab not in KEYPOINTS: continue
        side, vis = KEYPOINTS[lab]
        inside = [(p["box"][2] * p["box"][3], k) for k, p in people.items()
                  if p["box"][0] <= x <= p["box"][0] + p["box"][2] and p["box"][1] <= y <= p["box"][1] + p["box"][3] * 1.05]
        if inside:
            people[min(inside)[1]][f"{side}_ankle"] = (round(x, 2), round(y, 2), vis)
    return people, image


def load_frozen_v1(path):
    out = {}
    for t in json.load(open(path)):
        d = t["data"]
        if d.get("pass") != 1 or not t["annotations"]: continue
        ppl, _ = people_of(t["annotations"][-1]["result"])
        out[d["global_id"]] = list(ppl.values())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("export"); ap.add_argument("out")
    ap.add_argument("--annotation", choices=["last", "first"], default="last")
    ap.add_argument("--frozen-v1", default=None)
    a = ap.parse_args()
    v1ref = load_frozen_v1(a.frozen_v1) if a.frozen_v1 else {}
    cols = ["task_id", "global_id", "split", "split_v2", "pass", "task_kind", "selected_by", "contrast_group",
            "site_or_sequence_id", "annotator", "schema", "person_id", "x", "y", "w", "h",
            "support_surface", "support_v1", "on_scaffold", "hard_negative_type",
            "feet_on", "feet_visible", "proximity", "helmet_state", "harness_state", "anchor_visible",
            "height_band", "posture", "left_ankle", "right_ankle", "all_people_boxed", "image_flag",
            "v1_support", "flags"]
    n = 0
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for t in json.load(open(a.export)):
            d = t["data"]
            anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
            if not anns: continue
            ann = anns[-1] if a.annotation == "last" else anns[0]
            people, image = people_of(ann["result"])
            is_v2 = any("support_surface" in p for p in people.values()) or "all_people_boxed" in image
            for pid, p in people.items():
                row = {k: d.get(k, "") for k in ("global_id", "split", "split_v2", "pass", "task_kind", "selected_by",
                                                 "contrast_group", "site_or_sequence_id")}
                row.update(task_id=t.get("id"), annotator=ann.get("completed_by", ""), person_id=pid,
                           schema="v2" if is_v2 else "v1",
                           x=round(p["box"][0], 3), y=round(p["box"][1], 3), w=round(p["box"][2], 3), h=round(p["box"][3], 3),
                           all_people_boxed=strip("all_people_boxed", image.get("all_people_boxed", UNANNOTATED)),
                           image_flag=image.get("image_flag", ""))
                vals = {k: strip(k, p[k]) if k in p else UNANNOTATED for k in PERSON_FIELDS}
                if is_v2:
                    ss = vals["support_surface"]
                    row["support_surface"] = ss
                    row["support_v1"] = TO_V1_SUPPORT.get(ss, UNANNOTATED)
                    row["on_scaffold"] = ON_SCAFFOLD.get(ss, UNANNOTATED)
                    row["hard_negative_type"] = hard_negative_type(ss, vals["proximity"])
                    flags = consistency_flags(vals)
                else:
                    s = p.get("support", UNANNOTATED)
                    row["support_surface"] = UNANNOTATED   # v1 'none' can't be split into ground / other_elevated
                    row["support_v1"] = s
                    row["on_scaffold"] = V1_ON.get(s, UNANNOTATED)
                    flags = []
                for k in ("feet_on", "feet_visible", "proximity", "helmet_state", "harness_state", "anchor_visible",
                          "height_band", "posture"):
                    row[k] = vals[k] if is_v2 else (strip(k, p[k]) if k in p else UNANNOTATED)
                for side in ("left", "right"):
                    kp = p.get(f"{side}_ankle"); row[f"{side}_ankle"] = f"{kp[0]};{kp[1]};{kp[2]}" if kp else ""
                if is_v2 and d.get("global_id") in v1ref:
                    m = max(v1ref[d["global_id"]], key=lambda q: iou(q["box"], p["box"]), default=None)
                    if m is not None and iou(m["box"], p["box"]) >= 0.5:
                        row["v1_support"] = m.get("support", UNANNOTATED)
                        if row["support_v1"] not in (m.get("support"), UNANNOTATED):
                            flags.append("disagrees_with_v1")
                row["flags"] = "|".join(flags)
                w.writerow(row); n += 1
    print(f"wrote {n} people -> {a.out}")


if __name__ == "__main__":
    main()
