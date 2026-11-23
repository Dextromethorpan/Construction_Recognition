# Environments (one venv per toolset, Python 3.11 from Homebrew)

| venv | location | purpose | requirements |
|---|---|---|---|
| data | ~/GenIA-data/.venv | download, convert, dedup, split, export schemes (01_data) | requirements-data.txt (frozen) |
| labelstudio | ~/GenIA-data/.venv-ls | Label Studio app (02_labeling) | requirements-labelstudio.txt (frozen) |
| train | ~/GenIA-code/.venv-train | YOLO + PyTorch MPS (03_train_yolo, 05_eval, 06_quantize) | requirements-train.txt |
| vlm | ~/GenIA-code/.venv-vlm | MLX + Qwen2.5-VL (04_vlm) | requirements-vlm.txt |

The data and labelstudio venvs live next to the data because they were created first; venvs can't be moved
(their paths are baked in), so they stay where they are. Recreate any venv with:

    $(brew --prefix python@3.11)/bin/python3.11 -m venv <path> && <path>/bin/pip install -r envs/requirements-<name>.txt
