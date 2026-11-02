#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${BRIDGE_PYTHON:-/home/walt/bridge-rna/.venv/bin/python}"
export PYTHONPATH="$HERE/pipeline:$REPO/src" MPLBACKEND=Agg
mkdir -p "$HERE/results" "$HERE/work"
echo "Expected: ~15 min two-GPU TCGA inference; ~15 min analyses/IG. Logs are under $HERE/results."
"$PY" "$HERE/pipeline/tcga_representation.py" prepare 2>&1 | tee "$HERE/results/tcga_prepare.log"
"$PY" "$HERE/pipeline/tcga_representation.py" embed --start 0 --end 4908 --batch-size 4 --device cuda:0 2>&1 | tee "$HERE/results/tcga_embed_gpu0.log" & p0=$!
"$PY" "$HERE/pipeline/tcga_representation.py" embed --start 4908 --end 9816 --batch-size 4 --device cuda:1 2>&1 | tee "$HERE/results/tcga_embed_gpu1.log" & p1=$!
wait "$p0"; wait "$p1"
"$PY" "$HERE/pipeline/tcga_representation.py" combine "$HERE/work/tcga_embeddings_0_4908.npy" "$HERE/work/tcga_embeddings_4908_9816.npy"
"$PY" "$HERE/pipeline/tcga_representation.py" analyze 2>&1 | tee "$HERE/results/tcga_analysis.log"
"$PY" "$HERE/pipeline/transfer_and_heatmaps.py" 2>&1 | tee "$HERE/results/transfer_heatmaps.log"
"$PY" "$HERE/pipeline/de_gsea.py" 2>&1 | tee "$HERE/results/de_gsea.log"
"$PY" "$HERE/pipeline/integrated_gradients.py" --steps 8 --pathways 4 --device cuda:0 2>&1 | tee "$HERE/results/integrated_gradients.log"
"$PY" "$HERE/pipeline/ig_controls.py"
"$PY" "$HERE/pipeline/matched_organs.py" 2>&1 | tee "$HERE/results/matched_organs.log"
"$PY" "$HERE/pipeline/freeze_system.py"
"$PY" "$HERE/pipeline/test_invariants.py"
"$PY" "$HERE/pipeline/build_notebook.py"
(cd "$HERE" && "$PY" -m jupyter nbconvert --to notebook --execute bridge_hallmark_representation.ipynb --inplace --ExecutePreprocessor.timeout=600 > results/notebook.log 2>&1)
"$PY" "$HERE/pipeline/finalize.py"

