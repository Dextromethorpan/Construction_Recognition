# Experiments

One folder per study (`YYYYMMDD_<id>_<name>`): `config.yaml` (written before running), `arm_configs/`, run scripts, result tables and the verdict (`RESULTS.md` / `FINDINGS.md`). `RUNLOG.csv` lists every run.

Not included: feature caches (`*.npz`, `*.npy`), weights, contact sheets / images and any file over 2 MB.
The per-person score tables (`scores_*.csv`, `scores_cond_*.csv`, `xscores_*.csv`) are stored gzipped (`*.csv.gz`) to keep the repository small; `pandas.read_csv` reads them directly, or run `gunzip -k` first if a script expects the plain `.csv`.
Paths inside configs and result files were recorded on the author's machine and are shown as `~/GenIA-data/...` / `~/GenIA-code/...` (the code layout there is this repo's `code/` folder at the top level).
