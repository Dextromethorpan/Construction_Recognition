# Papers, part 2: relationships in general

Found second, by looking for papers with the same abstraction outside construction (a person supported by a surface, an object worn on the body, near vs on).

**Scene graphs and human-object interaction** (how relations are labeled and scored)
| Paper | Lesson we used |
|---|---|
| **Visual Genome** (2016) | Free-text relation labels are noisy and long-tailed; "on" dominates |
| **Neural Motifs** (2018) | Relations are guessable from the objects alone, so models cheat. Report a frequency baseline |
| **Unbiased SGG** (Tang 2020) | Use mean recall per predicate, not plain recall |
| **PSG / OpenPSG** (2022) | Masks and a small, defined predicate list make relations learnable. Hence our short vocabulary plus "unclear" |
| **HICO-DET** (2018), **V-COCO** (2015) | "Stand on", "wear" and `no_interaction`; a pair counts only if both boxes and the verb are right |
| **UPT** (2022), **CDN** (2021) | Freeze a detector and learn only the relation head. Our detector + judge design |

**Support and contact** (the same abstraction as ours: a person supported by a surface)
| Paper | Lesson we used |
|---|---|
| **Silberman et al. 2012** (support inference from RGB-D) | "What supports what" from below. Physical reasoning beats 2D rules, and region-finding errors cost about 20 points, just like our detector misses |
| **HOT** (Chen 2023) | Contact maps per body part, such as the feet |
| **PaStaNet / HAKE** (2020) | Body-part labels ("foot treads on") help most with few examples (+13.9 mAP one-shot vs +6.4 full). Basis for `feet_on` |
| **TIN** (2019) | First decide "interacting or not", then which relation. Basis for the `proximity` hard-negative field |
| **SpatialSense** (2019), **Rel3D** (2020) | Box position alone almost solves "on", so label hard negatives and **contrast pairs** on purpose (same site, one person on, one in front) |
| **Siebert & Lin 2020** (motorcycle helmets) | The twin problem outside construction: person on a vehicle plus helmet per person, reported as a rate per site |

**VLMs and spatial reasoning** (why we did not rely on a VLM)
| Paper | Lesson we used |
|---|---|
| **What'sUp** (2023) | CLIP-style models are at chance on on/under/in front. Fine-tuning on noisy automatic relation labels made it **worse**. So the test set must be checked, not machine-made |
| **VSR** (2023) | Humans 95%, models about 70% on spatial relations; drop items labelers disagree on |
| **Roboflow100-VL** (2025) | Zero-shot VLM detection is weak (Qwen2.5-VL-72B 5.6 mAP); 10 labeled shots doubled Grounding DINO. Labels still matter |
| **Set-of-Mark** (2023), **SpatialRGPT** (2024) | If a VLM is used, point at numbered regions and add depth |
| **LLM4SGG** (2024), **SpatialVLM** (2024) | Cheap machine-made training labels are fine, but the test set stays human-checked |

**Label-efficiency papers** (what to do with few labels)
| Paper | Lesson we used |
|---|---|
| **Niitani 2019**, **Co-mining 2021**, **Background Recalibration Loss 2020** | Unlabeled objects get taught as background. Our 16% → 55% fix |
| **Unbiased Teacher** (2021) | A teacher labels unlabeled images for a student. Tried in the post-test round with CS10k person boxes: rejected for us, because the teacher missed the same people |
| **RLIPv2** (2023), **MX-HOI**, **HOI-Synth** | No paper gives a "magic number" of relation labels. HOI-Synth also warns that simply mixing synthetic images does not help, which matched our own result |

