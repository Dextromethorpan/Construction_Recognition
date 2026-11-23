# Construction Recognition: helmet, harness and "on a scaffold" from site photos

For every person in a construction photo, this project answers three questions:
- **Helmet:** is the person wearing one?
- **Harness:** is the person wearing one?
- **On a scaffold:** is the person standing on or climbing a scaffold?

Everything was built on a MacBook Air M5 (16 GB), using public datasets only. The first two questions are about objects on a person. The third is a **relationship** between a person and a structure, and it is the hard part.

## Results at a glance
**The pipeline:** a YOLO11n detector (512 px) finds people. Small per-person heads then read features from a frozen image encoder:
- the **on-scaffold judge** sees a wide crop around the person
- the **helmet and harness heads** see a tight crop

The deployment build is **CoreML INT8: 2.86 MB, 4.1 ms per photo**.

**One-time test** (336 real photos, opened once after every choice was fixed on validation):
| Subset | On-scaffold recall | False alarms | Helmet bal. acc. | Harness bal. acc. |
|---|---|---|---|---|
| All real photos | **0.517** [0.43–0.61] | 5.4% (target 1.5%) | 0.734 | 0.779 |
| In-distribution | 0.805 | 8.9% | 0.730 | 0.806 |
| CS10k (other domain) | 0.176 | 4.5% | 0.667 | 0.704 |
| Held-out source | 0.072 | 0% | 0.635 | 0.688 |
| All real, CoreML INT8 | 0.521 | 5.2% | 0.734 | 0.779 |

**Main finding:** recall transfers from validation to test, but the false-alarm rate does not. On unfamiliar photos, most misses are people the detector never boxes. The full story is in [docs/06_results.md](docs/06_results.md).

## Repository map
| Folder | What |
|---|---|
| `docs/01_dataset.md` | Data card: sources, licences, dedup, split by site, counts |
| `docs/02_labels.md` | How the labels were made (two AI labelers, cross-checked) and how the rules changed |
| `docs/03_model.md` | The final model, the models tried, and what other models would likely do |
| `docs/04_loops.md` | The data, labeling and controlled-experiment loops, and how Claude ran them |
| `docs/05_quantization.md` | FP16 / INT8 exports, how they were measured, results |
| `docs/06_results.md` | Every experiment, its conclusion, and the path to the final model |
| `docs/logs/` | The original working logs (data, labeling, ablation, results) that every number in the docs comes from, with their charts |
| `docs/papers/` | Papers used: construction first, then the more abstract relationship papers |
| `experiments/` | One folder per study: config written before running, results, verdict. `RUNLOG.csv` has every run |
| `data/` | Split manifests, checksums, our relation labels (v1, v2), guidelines. No images (label schemes: `code/configs/label_schemes/`) |
| `demo/` | `demo.py`, saved heads and example outputs for the three cases |
| `models/` | The detector (CoreML INT8 and PyTorch weights) and the frozen encoder used by the heads |
| `code/` | Data pipeline, labeling tools, training, ablation, controlled runner, quantization |
| `envs/` | Pinned requirements per virtual environment |

## Run the demo
```bash
python -m venv .venv && .venv/bin/pip install -r envs/train-requirements.lock.txt
.venv/bin/python demo/demo.py path/to/photo_or_folder --out demo_out/
```
Each person gets a box (green = on a scaffold) and a tag with the helmet, harness and scaffold answers and their scores. The CoreML model needs macOS. On other systems, use `models/detector_fp32.pt`.

## Rebuild the data
The images are not included. The CS10k test images come from a gated dataset, and the other sources are large. `code/01_data/` downloads, converts, deduplicates and splits them, and `data/` holds the manifests and SHA256 checksums to verify the result. See [docs/01_dataset.md](docs/01_dataset.md).

## Disclosures
- **The labels are AI-made.** Claude labeled the relation data. Codex (OpenAI) labeled 15% blind to measure agreement and reviewed the rest. No human audit has been done yet.
- **The early exploration looked at test results** (docs/06_results.md, chapter 5). Every choice after it was made on validation under a rule written in advance.
- **The test set was opened once**, and that is enforced in code (`code/08_controlled/runlog.py`). The test uses v1 labels, while validation uses v2 labels.
- **The post-test results are validation only.** This includes the helmet head-crop improvement and the 120 extra photos, which were labeled by Claude only. They have no test number.
- **How it was built:** the work was carried out by Claude (Anthropic) as a coding and labeling agent, under decisions, rules and approvals set by the author.

## Licence
The code builds on Ultralytics YOLO (AGPL-3.0), so this repository is released under **AGPL-3.0**. Dataset licences are listed per source in `docs/01_dataset.md`. Our labels may be used for research and evaluation, with attribution.
