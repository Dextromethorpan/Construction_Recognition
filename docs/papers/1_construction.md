# Papers, part 1: construction and PPE

Found first, by searching for construction-safety work. Links are in `docs/papers/links.md`.

| Paper | What it does | What we took from it |
|---|---|---|
| **ConstructionSite-10k** (Chen & Zou 2025, "Are large pre-trained VLMs effective construction safety inspectors?") | 10,013 site photos with captions, rule violations and violator boxes; tests GPT-4o, LLaVA and others | Became our **external test set (CS10k)**. Its finding that VLMs have high recall but low precision and weak grounding supported choosing YOLO + a judge over a VLM |
| **Safe-Construct** (Chharia et al. 2025) | Pose keypoints, multi-view 3D, then rules on distances; "worker on platform" = feet-to-platform distance | The **feet** are the signal for "on". Led to ankle keypoints and `feet_on` in v2 |
| **Jin, Xu & Xie 2026** (scaffold worker risk) | YOLO + pose + SAM2 scaffold mask; climbing = body keypoints overlapping the mask; height band | Platform as a polygon or mask, height band, posture tags |
| **Park et al. 2023** (ladder workers) | YOLO then rules on working height | Geometry plus rules can replace relation labels |
| **Yi, Xu & Tan 2026** (PPE at height with scene graphs) | Detects lifeline and hook, matches a scene graph against a rule graph from regulations | Tie-off as a relation needs lifeline and hook objects; keep violations as rules |
| **Liu et al. 2023** (scene graphs + regulation text) | Image triples matched to regulation triples | **Label facts, compute violations** |
| **SafeSceneReason** (2026) | Safety scene graphs; PPE as missing / present / **unknown** | PPE needs an explicit "not visible" state |
| **Fang et al. 2018** (harness detection) | Find the worker, then classify the crop for a harness | Our exact two-stage design. Their 80% harness precision shows false alarms are the known weak spot |
| **Wójcik et al. 2021** (hard hat by head keypoint) | Helmet "worn" if the head point is inside a helmet box; no "wearing" labels | Geometry beats a learned relation when labels are few; supports the later head crop for helmet |
| **SHEL5K** (Otgonbold et al. 2022) | Relabeled a helmet dataset that lacked person boxes | Precedent for our missing-person fix (qualitative only, as Claude's review pointed out) |
| **Kim et al. 2020** (active learning on construction sites) | Label only the images the model is unsure about | 180 chosen images matched 720 random ones. Basis for `selected_by: active` |
| **SODA** (Duan et al. 2022) | Construction detection with scaffold, person, helmet, hook | Wanted, but only on Baidu Netdisk, so skipped |
| **Sahraoui 2025** (hazards with LLMs and VLMs) | Small open VLMs on CS10k | Small VLMs can beat GPT-4 on PPE rules; a possible future cascade |

**The gap we can claim:** no paper labels "person on scaffold" as a relationship in still photos. The scaffold AI papers check the structure, not the worker on it.

