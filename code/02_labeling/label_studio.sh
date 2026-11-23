#!/bin/zsh
# Start Label Studio locally with access to ~/GenIA-data images (nothing is uploaded).
# One-time install (separate venv so it can't break the data venv):
#   $(brew --prefix python@3.11)/bin/python3.11 -m venv ~/GenIA-data/.venv-ls && ~/GenIA-data/.venv-ls/bin/pip install label-studio
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$HOME/GenIA-data"
export LABEL_STUDIO_BASE_DATA_DIR="$HOME/GenIA-data/labeling_queue/label_studio_db"
exec "$HOME/GenIA-data/.venv-ls/bin/label-studio" start --port 8081 --no-browser
