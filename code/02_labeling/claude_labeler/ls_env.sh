# Run a python file inside Label Studio's Django context: ls_env.sh script.py [args via env]
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$HOME/GenIA-data"
export LABEL_STUDIO_BASE_DATA_DIR="$HOME/GenIA-data/labeling_queue/label_studio_db"
cd "$HOME/GenIA-data/.venv-ls/lib/python3.11/site-packages/label_studio"
"$HOME/GenIA-data/.venv-ls/bin/python" manage.py shell -c "exec(open('$1').read())" 2>&1 | grep -v -E "RuntimeWarning|warnings.warn|objects imported|pypi.org"
