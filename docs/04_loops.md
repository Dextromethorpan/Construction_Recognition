# The loops: how the experiments were run

The experiments were run by Claude (Anthropic) as an agent on the author's Mac, under rules the author set and approved step by step. Codex (OpenAI) acted as the second, independent labeler and reviewer.


The project was never a straight line. Five loops ran, each with its own stopping rule, and the outer one fed the inner ones.

## Loop 1. Data Acquisition Agent (collect → convert → dedup → split → check)
- Written as a prompt with phases, **stopping rules** and **done checks** (*Data Acquisition Agent*). It never labels anything itself and stops at checkpoints for your OK.
- **It worked:** it stopped at the split when only 123 images had people and scaffolds together. That triggered adding five scaffold datasets (1,990 images), and only then was split v1 frozen. Every step is in *agent_log*.

## Loop 2. Labeling by two AI labelers, supervised (label → blind check → review → measure)
1. **Claude, first pass:** renders each image with a percentage grid, writes a JSON spec per task and pushes it into Label Studio. Then it rechecks helmet and harness for all 1,359 people on zoomed crops (about 900 values corrected).
2. **Codex, blind second pass** on 114 images (15%), without seeing Claude's answer. Cohen's kappa between the two is the label-quality estimate.
3. **Codex review** of all 754 images: 716 fixed, 38 accepted.
4. **Claude supervises Codex** through a script (`supervise.sh`): one task at a time with the image attached, 4 then 8 parallel workers, a watchdog and auto-retry after an internet drop, a `STOP` file to pause, and every run logged. Claude spot-checks Codex by drawing its specs on the image.
- **What it produced:** the label-quality numbers (support 0.69, helmet 0.80, harness 0.39) that told us which rule to fix next.

## Loop 3. The labeling optimization loop (v2)
```
Label → Train (fixed recipe, 5 seeds) → Threshold on val → Evaluate with the decision rule
  ↑                                                                   ↓
Change guideline / choose what to label  ←  Error analysis (train and val only)
exit: the rule says stop → test once → report
```
- Every guideline change gets a version number and a reason, and relabeling is applied to train and val alike.
- Each image records why it was chosen (`selected_by`: random, active, contrast, legacy), which controls selection bias.
- Fields with kappa below 0.6 are not used as targets.
- **What it produced:** labels v2 (3,451 people) and the harness rule fix (0.39 → 0.66).

## Loop 4. The controlled experiment loop (hypothesis → config → 5 draws → rule → log)
1. Write the **hypothesis and config first** and save them before the first run. The config hash is written on every log row.
2. Train each arm with **5 draws** (seeds 0 to 4) using the same recipe.
3. Choose the **cutoff on validation only** (1.5% false alarms).
4. Apply the **decision rule**: ≥ 0.05 gain and a confidence interval that excludes zero. Ties go to the simpler option.
5. Log **every run**, including failed and abandoned ones, in `RUNLOG.csv`: 159 rows by the end, nothing deleted.
6. **Stop rule:** no arm passes in consecutive studies → stop and test.
- The tooling enforces it: `runlog.py` refuses a test run without a written decision, and refuses a second test run. `threshold.py` refuses to choose a cutoff on test rows.
- **What it produced:** C1 adopted, four ideas rejected, the stop rule met after three studies in a row, and one honest test.

## Loop 5. Error analysis chooses the next hypothesis
This is the loop that connects all the studies. Each study's idea came from the previous study's errors.
- Misses split into "never boxed" vs "boxed but judged wrong" → the detector study.
- Missed people turned out to be crowded, behind tubes, low contrast (not just tiny) → lower confidence and looser NMS.
- Once more people were boxed, the judge confused decks and rebar with scaffolds → the next step is support labels, not a bigger detector.
- Helmet errors were mostly false "no helmet" → the head crop, confirmed twice.

## The loop for GenIA in production (future)
Deploy → the model flags uncertain people → a human labels them → retrain under the same decision rule → redeploy. Every new site improves the model, and the decision rule keeps it from getting worse silently.


## How Claude executed the loops in practice
- **Where it ran:** everything ran locally on a MacBook Air M5 (16 GB, MPS GPU) in project virtual environments (`envs/`). Claude worked through a shell on the Mac. The author approved every step that installed software, changed data or opened the test set.
- **One folder per study:** `experiments/<date>_<name>/` holds the `config.yaml` (hypothesis, arms, measures, decision rule), written and saved **before** the first run, the run scripts, and `RESULTS.md` with the verdict.
- **The run log:** `code/08_controlled/runlog.py` appends one row per run to `experiments/RUNLOG.csv` with the study, arm, seed, config hash, metrics and status (`run`, `failed`, `abandoned`, `decision`, `test`). Rows are never deleted.
- **Guards in code, not just in rules:** `runlog.py check-test` refuses a test run without a `decision` row and refuses a second test run for the same study. `threshold.py` refuses to choose a cutoff on test rows.
- **The judge and the comparison:** `judge.py` caches detections and crop features once per detector, fits the per-person heads for 5 draws (80% of the training images, seeds 0–4), and writes scores. `compare.py` computes the paired bootstrap difference (2,000 resamples by image, cutoff re-chosen inside each resample) and applies the decision rule.
- **Labeling automation:** Claude rendered each image with a percentage grid, wrote one JSON spec per task and pushed it into Label Studio. Codex was driven by `supervise.sh` (one task per call with the image attached, 4 then 8 parallel workers, a watchdog and retries, a `STOP` file to pause). Agreement was computed with `kappa.py` / `kappa_v2.py`.
- **Long runs:** detector training took about 1.6 hours (6.4 min per epoch at 512 px). Runs were kept awake with `caffeinate`, and the laptop lid had to stay open (one overnight run stopped when the Mac slept).
- **Reporting:** after each study Claude updated the experiment log and the results notes, with charts, so every decision has a written reason.
