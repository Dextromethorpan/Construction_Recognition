"""Single source of truth for relation v2 field names and values (GUIDELINES v2.0, D_rich_v2.yaml).
Used by parse_v2.py, kappa_v2.py, build_queue.py. v1 code in claude_labeler/ is not changed."""

UNANNOTATED = "unannotated"   # a field that was never set. NEVER treated as "no".

SUPPORT_SURFACE = ["scaffold_platform", "scaffold_frame_or_attached_ladder", "ground", "other_elevated", "unclear"]

# Per-person choice fields: name -> (Label Studio prefix, values without prefix)
PERSON_FIELDS = {
    "support_surface": ("", SUPPORT_SURFACE),
    "feet_on": ("feet:", ["platform", "frame_or_ladder", "ground", "other_surface", "not_visible"]),
    "feet_visible": ("fv:", ["both", "one", "none"]),
    "proximity": ("prox:", ["in_front_of", "behind", "under", "next_to", "far", "no_scaffold_in_image"]),
    "helmet_state": ("helmet:", ["worn", "not_worn", "held", "not_visible"]),
    "harness_state": ("harness:", ["worn", "not_worn", "not_visible"]),
    "anchor_visible": ("anchor:", ["yes", "no"]),
    "height_band": ("h:", ["ground", "low", "mid", "top", "unknown"]),
    "posture": ("", ["standing", "climbing_posture", "kneeling", "sitting", "leaning"]),
}
REQUIRED_ALWAYS = ["support_surface", "feet_on", "feet_visible", "helmet_state", "harness_state"]
IMAGE_FIELDS = {"all_people_boxed": ("boxed:", ["yes", "no"])}

# Derived on_scaffold (C_relation_v2.yaml)
ON_SCAFFOLD = {"scaffold_platform": "on", "scaffold_frame_or_attached_ladder": "on",
               "ground": "not_on", "other_elevated": "not_on", "unclear": "unclear"}
# Backward map v2 -> v1 support, so v1 and v2 can be scored together
TO_V1_SUPPORT = {"scaffold_platform": "stands_on", "scaffold_frame_or_attached_ladder": "climbing",
                 "ground": "none", "other_elevated": "none", "unclear": "unclear"}
# v1 support -> on_scaffold (C_relation_v1.yaml: none -> next_to, renamed not_on here)
V1_ON = {"stands_on": "on", "climbing": "on", "none": "not_on", "unclear": "unclear"}

# Fields with these values need proximity
NEEDS_PROXIMITY = {"ground", "other_elevated", "unclear"}

KEYPOINTS = {"left_ankle": ("left", 2), "right_ankle": ("right", 2),
             "left_ankle_occluded": ("left", 1), "right_ankle_occluded": ("right", 1)}


def strip(field, value):
    """'feet:ground' -> 'ground'. Unknown values are returned unchanged (and flagged by the caller)."""
    prefix = PERSON_FIELDS.get(field, IMAGE_FIELDS.get(field, ("", None)))[0]
    return value[len(prefix):] if prefix and value.startswith(prefix) else value


def with_prefix(field, value):
    prefix = PERSON_FIELDS.get(field, IMAGE_FIELDS.get(field, ("", None)))[0]
    return prefix + value


def hard_negative_type(support_surface, proximity):
    if ON_SCAFFOLD.get(support_surface) != "not_on":
        return ""
    return f"{support_surface}+{proximity}"


def consistency_flags(p):
    """p: dict of stripped per-person values. Returns a list of review flags (never auto-fixes)."""
    f = []
    ss, fo, fv = p.get("support_surface", UNANNOTATED), p.get("feet_on", UNANNOTATED), p.get("feet_visible", UNANNOTATED)
    for k in REQUIRED_ALWAYS:
        if p.get(k, UNANNOTATED) == UNANNOTATED:
            f.append(f"missing_{k}")
    if ON_SCAFFOLD.get(ss) == "on" and fo == "ground":
        f.append("on_scaffold_but_feet_ground")
    if ss == "ground" and fo in ("platform", "frame_or_ladder"):
        f.append("ground_but_feet_on_scaffold")
    if fv == "none" and fo not in ("not_visible", UNANNOTATED):
        f.append("feet_not_visible_but_feet_on_set")
    if ss in NEEDS_PROXIMITY and p.get("proximity", UNANNOTATED) == UNANNOTATED:
        f.append("missing_proximity")
    if p.get("harness_state") == "worn" and p.get("anchor_visible", UNANNOTATED) == UNANNOTATED:
        f.append("missing_anchor_visible")
    return f
