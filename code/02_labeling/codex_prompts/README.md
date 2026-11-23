# Codex labeling prompts (GenIA relation v1)

Each file in `pass2/` and `review/` is a self-contained prompt for one Label Studio task.
Run them one at a time. Each prompt says exactly which task to touch.

- `pass2/task_NNNN.md`: label one blind second-pass task (114 in total). These measure agreement with Claude's first pass.
- `review/task_NNNN.md`: check Claude's first-pass labels on one task and fix what is wrong (754 in total).
- `CONTEXT.md`: shared rules, label schema, spec format and commands. Every prompt tells you to read it first.

Progress is tracked in `~/GenIA-data/labels/relations/v1/codex_labeled_tasks.csv` (written automatically) and
`~/GenIA-data/labels/relations/v1/codex_reviews/NNNN.json` (one per reviewed task).
