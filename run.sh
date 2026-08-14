set -euo pipefail
PIPELINE_DIR="${NBA_PIPELINE_DIR:-$(pwd)/pipeline}"
export NBA_MODELS_DIR="${NBA_MODELS_DIR:-$(pwd)/models}"

# Convert to Windows path form and use ';' separator for Windows Python
if command -v cygpath >/dev/null 2>&1; then
    PIPELINE_DIR_WIN="$(cygpath -w "$PIPELINE_DIR")"
else
    PIPELINE_DIR_WIN="$PIPELINE_DIR"
fi
export PYTHONPATH="${PIPELINE_DIR_WIN};${PYTHONPATH:-}"

echo "Pipeline modules : ${PIPELINE_DIR}"
echo "Model artifacts  : ${NBA_MODELS_DIR}"
echo "Serving on       : http://0.0.0.0:8000  (frontend at /, docs at /docs)"
exec uvicorn nba_api.main:app --host 0.0.0.0 --port 8000 "$@"