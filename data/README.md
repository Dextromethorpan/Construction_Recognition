# Data (no images)

- `splits/v1`, `splits/v2`: split id lists, manifests, `SHA256SUMS`, build report. v1 is the original split (used by the one-time test, `test_*.txt`); v2 is the relation split used for training/validation. `splits/v1/learning_curve/` holds the learning-curve subsets.
- `labels/relations/v1`: frozen v1 relation labels (`codex_reviewed.json`, task tables, per-image spec archives `*_specs.tar`, `synthetic_images.csv`, `SHA256SUMS`).
- `labels/relations/v2`: frozen v2 relation labels (`labels_v2.json`, `people_v2.csv`, review status, kappa tables, `SHA256SUMS`).
- `labels/relations/v2x/val_extra_people.csv`: the 120 extra validation photos (Claude-labeled only; validation use). Image paths are relative to the data root (`extracted/...`).
- `labeling/relation_v1`, `labeling/relation_v2`: labeling guidelines and the Label Studio config.
- Label schemes live in `code/configs/label_schemes/`.

Images are not redistributed (CS10k is gated; other sources are large). Rebuild them with `code/01_data/` and verify with the `SHA256SUMS` files.
