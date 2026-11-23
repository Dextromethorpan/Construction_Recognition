# Experiments and results: the full story

Every number comes from `experiments/*/RESULTS.md` and `experiments/RUNLOG.csv`.


## The question
From a construction photo, find every person and answer three questions about each one: is the person wearing a **helmet**, wearing a **harness**, and **standing on or climbing a scaffold**? It had to run on a MacBook Air M5 with 16 GB, with public datasets only and no labeled data of our own at the start.

The first two questions are about **objects** (is there a helmet on this head?). The third is about a **relationship** between a person and a structure. That difference shapes the whole story: helmet and harness behaved like normal detection problems, and "on a scaffold" turned out to be the hard part every time.

## Chapter 1. Decide the approach before building
- **Choice:** a small YOLO detector plus per-person judgements, not a big vision-language model. A 7B VLM in 4 bits was the maximum the Mac could hold, published benchmarks showed zero-shot VLMs are weak at this kind of detection, and YOLO can be quantized and run in milliseconds.
- **Conclusion:** the relationship needs its own labels. No public dataset labels "person on scaffold" as a relation, so we would have to make them.

## Chapter 2. Build the data (Data Acquisition Agent)
- 13 public sources were collected, converted to one format and deduplicated (9,662 near-duplicates removed).
- The split is **by source, not by photo**, so the same site never appears in both training and test. One whole source (rf_harness_mks) was held out completely, and **CS10k** became an external test set from a different domain. Split v1 was frozen with checksums.
- **Conclusion:** the agent stopped itself when only 123 images had people and scaffolds labeled together (below its 200 threshold). Adding five scaffold datasets closed the gap to 1,990 images. The lesson was that the stopping rules caught a real data gap before any training.

## Chapter 3. Make the relationship labels (AI labelers, cross-checked)
- Claude labeled 754 images in Label Studio. Codex labeled 114 of them blind, so we could measure agreement, and then reviewed all 754.
- Agreement (Cohen's kappa) was strong for helmet (0.80) and good for "on scaffold" (0.69), but **weak for harness (0.39)** and useless for "tied off" (about 0), because the two labelers used different conventions for hidden straps.
- Codex added about 311 people Claude had missed: 1,670 people in total.
- **Conclusion:** label agreement is the ceiling for any model, so it has to be measured, not assumed. The harness rule needed fixing.

## Chapter 4. First training: compare labeling policies (ablation)
- Six labeling policies were tried with the same model and seed (drop or keep "unclear", harness from boxes or from the per-person field, Claude or Codex labels, with or without synthetic images).
- Every policy landed between 0.55 and 0.57 F1 on scaffold. That is inside the noise.
- **Conclusion:** the policy barely matters, because the variants changed only about 1% of the detector's data. We kept the best options: drop unclear, take harness from the per-person field (0.70 → 0.75), use Codex-reviewed labels.
- **The real finding was elsewhere:** the detector found only **16%** of the people.

## Chapter 5. Fix person finding, discover the real bottleneck
- The source datasets drew boxes on only some people, so the detector learned "an unboxed person is background". A ready-made COCO person detector filled in 34,459 missing boxes.
- People found: **16% → 55%**, with no cost to helmet or harness.
- But "on scaffold" F1 **did not move** (0.51 to 0.57 in every setup). The newly found people were mostly on the ground.
- **Conclusion:** the scaffold judgement was **data-starved, not model-starved**. It learned from only 88 real people (63 on a scaffold, 18 not on), because most labeled people sat in the test set.
- **Honest note:** this exploration looked at test results while choosing. That is why everything after it was redone under strict rules.

## Chapter 6. Research review, then labels v2
- Codex reviewed the research on relationship labeling, and Claude checked that review. It found no "magic number" of labels, a circular comparison (Codex scored against its own labels), and the key practical point: put the labels where training can use them.
- **Labels v2:** 1,189 images and 3,451 people. Real on-scaffold people in training went from 63 to 245, and not-on people from 18 to 1,638.
- A clearer harness rule ("not worn" only when chest and shoulders are visible, otherwise "not visible") took harness agreement from **0.39 to 0.66**.
- Fields under kappa 0.6 (proximity 0.35) are not used as targets.
- **Conclusion:** fix the data and the rules before tuning the model.

## Chapter 7. Controlled experiments (validation only, rule written first)
The rule: a change replaces the baseline only if it adds **at least 0.05** on-scaffold recall at 1.5% false alarms (mean of 5 training draws) **and** the 95% confidence interval of the gain excludes zero. The test set stays closed.

| Study | Idea | Result | Decision |
|---|---|---|---|
| Free checks | Where do the misses come from? | 56% of misses are people the detector never boxed | Points to context and detector |
| Context C0–C3 | Show the judge a bigger crop around the person | **0.45 → 0.53 (+0.08)** | **Adopted (C1)** |
| C2 structure | Add "are the feet inside a scaffold box?" | +0.012 | Not adopted, the big crop already sees the scaffold |
| D1 detector | 960 px, bigger model, 1280 px, tiling | Best +0.039, not significant (the info-only tiled arm reached +0.053, also not significant) | Keep yolo11n 512 (960 px is the runner-up) |
| H1 harness crop | Tight torso crop for the harness head | −0.040 | Not adopted, it loses shoulder and lanyard context |
| Q1 quantization | FP16 and INT8 exports | **CoreML INT8: 2.86 MB, 4.1 ms, no accuracy loss** | **Adopted** |

- **Conclusion:** one real gain (seeing context around the person), one free win (quantization), and three ideas rejected in a row. That met the stop rule.

## Chapter 8. Open the test set, once
With your OK on 21 Nov at 20:03, the frozen pipeline was run on the 336 test photos.

| Subset | On-scaffold recall | False alarms |
|---|---|---|
| **All real photos** | **0.517** [0.43–0.61] | 5.4% (target 1.5%) |
| In-distribution | 0.805 | 8.9% |
| CS10k (other domain) | 0.176 | 4.5% |
| Held-out source | 0.072 | 0% |
| CoreML INT8 | 0.521 | 5.2% |

Helmet balanced accuracy 0.73 and harness 0.78 on all real photos.

- **Conclusions:** recall transferred (0.52 test vs 0.53 validation), but the false-alarm rate tripled. On familiar photos the pipeline finds 8 of 10 people on scaffolds. On unfamiliar photos it fails mostly because **the detector never boxes the people**: 22 of 42 on CS10k, 33 of 39 on the held-out source. Quantization cost nothing on test either.

## Chapter 9. After the test (validation only, reported separately)
The test set is used up, so everything here has **no test number** and is presented as validation evidence only.
- A fine-tuned judge is no better (0.515 vs 0.534). The judge is not the limit on normal photos.
- The false-alarm jump is a **domain shift**, not a noisy cutoff: across 500 validation half-splits the cutoff holds about 1.6%.
- Machine-made person boxes for CS10k did not help (the teacher misses the same people).
- 960 px + lower confidence + looser NMS boxes many more people (139 vs 106 of 200 on new photos), **but** on hard scaffold scenes the judge then becomes the limit: people on decks and rebar next to scaffolds look the same. Not adopted.
- **Helmet head + zoomed head crop: confirmed twice on new data** (+0.133 and +0.077 balanced accuracy). The main effect is fewer false "no helmet" alarms (80% → 92% correct). Adopted.
- A feet/support crop for the judge gave almost nothing.

## Where the journey ends (the one-minute version)
1. Helmet and harness are solved reasonably well. "On scaffold" is a relationship and is the hard part.
2. Each bottleneck we removed revealed the next: **missing person boxes → too few training people → inconsistent labels → too little context → detector misses on new domains → on hard scenes, the judge's understanding of support.**
3. Everything after the exploration was decided on validation with a rule written in advance, and the test was opened once.
4. **Next steps:** more person boxes and support labels from new domains (feet and surface), the 960 px detector, and a bigger validation set (118 positives give ±0.06, too wide to confirm a 5-point gain; about 250 to 300 are needed).

