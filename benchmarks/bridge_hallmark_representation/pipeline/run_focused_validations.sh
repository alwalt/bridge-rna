#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BRIDGE_PYTHON:-/home/walt/bridge-rna/.venv/bin/python}"
export PYTHONPATH="$HERE/pipeline:$REPO/src" MPLBACKEND=Agg
mkdir -p "$HERE/results/ig_validation" "$HERE/results/pan_cancer_gsea"

echo "Expected: ~50 min two-GPU 128-step IG validation; ~1 min for seven pan-cancer GSEA contrasts."
"$PY" "$HERE/pipeline/cache_ig_gradients.py" --device cuda:0 --sample-indices 0 2 2>&1 | tee "$HERE/results/ig_cache_gpu0.log" & p0=$!
"$PY" "$HERE/pipeline/cache_ig_gradients.py" --device cuda:1 --sample-indices 1 3 2>&1 | tee "$HERE/results/ig_cache_gpu1.log" & p1=$!
wait "$p0"; wait "$p1"
"$PY" "$HERE/pipeline/validate_integrated_gradients.py" 2>&1 | tee "$HERE/results/ig_validation.log"
"$PY" "$HERE/pipeline/ig_controls.py" --results-dir ig_validation
"$PY" "$HERE/pipeline/pan_cancer_gsea.py" 2>&1 | tee "$HERE/results/pan_cancer_gsea.log"
"$PY" "$HERE/pipeline/test_invariants.py"
"$PY" "$HERE/pipeline/build_notebook.py"
(cd "$HERE" && "$PY" -m jupyter nbconvert --to notebook --execute bridge_hallmark_representation.ipynb --inplace --ExecutePreprocessor.timeout=600 > results/notebook.log 2>&1)
"$PY" "$HERE/pipeline/finalize.py"
