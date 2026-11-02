#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BRIDGE_PYTHON:-/home/walt/bridge-rna/.venv/bin/python}"
export PYTHONPATH="$HERE/pipeline:$REPO/src"
export MPLBACKEND=Agg
mkdir -p "$HERE/results" "$HERE/work"
echo "Expected: ~5 minutes for reusable GTEx Parts I-II; contextual phases depend on cohort and GPU (~seconds/sample)."
"$PY" "$HERE/pipeline/audit.py" 2>&1 | tee "$HERE/results/audit.log"
"$PY" "$HERE/pipeline/sample_space.py" 2>&1 | tee "$HERE/results/run.log"
"$PY" "$HERE/pipeline/build_notebook.py"

