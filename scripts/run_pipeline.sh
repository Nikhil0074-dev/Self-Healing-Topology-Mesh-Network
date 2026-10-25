#!/usr/bin/env bash
# End-to-end: tests -> dataset -> train -> experiments -> report.
set -euo pipefail
cd "$(dirname "$0")/.."

# Override with PIPELINE_PYTHON when a virtual environment or a specific
# interpreter is required. WSL commonly provides python3 rather than python.
if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
    PYTHON_BIN="$PIPELINE_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v py.exe >/dev/null 2>&1; then
    # Prefer the Windows launcher in WSL when project packages are installed
    # in the Windows Python environment.
    PYTHON_BIN="$(command -v py.exe)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python was not found. Install Python 3 or set PIPELINE_PYTHON." >&2
    exit 127
fi

if ! "$PYTHON_BIN" -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
    echo "Cannot execute Python interpreter: $PYTHON_BIN" >&2
    exit 127
fi

echo "Using Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" -m pytest tests -q
"$PYTHON_BIN" -m gnn.dataset --out experiments/results/dataset.pt \
    --train "${DATASET_TRAIN:-60}" --val "${DATASET_VAL:-12}" \
    --test "${DATASET_TEST:-12}" --unseen "${DATASET_UNSEEN:-12}" \
    --steps "${DATASET_STEPS:-300}"
"$PYTHON_BIN" -m gnn.train --data experiments/results/dataset.pt --out models/link_gnn_v1.pt \
    --epochs "${TRAIN_EPOCHS:-30}"
"$PYTHON_BIN" -m simulator.experiment_runner --seeds "${SEEDS:-10}" --workers "${WORKERS:-1}" --out experiments/results/results.csv
"$PYTHON_BIN" -m experiments.analyze experiments/results/results.csv --out experiments/results/report.md
