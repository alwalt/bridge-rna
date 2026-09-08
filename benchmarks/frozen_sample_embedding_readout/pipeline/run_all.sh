#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=benchmarks/frozen_sample_embedding_readout/results
mkdir -p "$OUT/workers"
PY=.venv/bin/python
SCRIPT=benchmarks/frozen_sample_embedding_readout/pipeline/run_benchmark.py

echo "[$(date '+%F %T')] estimated wall time: 3-5 hours on two RTX 3090 GPUs" | tee "$OUT/run.log"
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" --phase train --folds 0 2 4 --device cuda:0 \
  > "$OUT/workers/gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPT" --phase train --folds 1 3 --device cuda:0 \
  > "$OUT/workers/gpu1.log" 2>&1 &
P1=$!
while kill -0 "$P0" 2>/dev/null || kill -0 "$P1" 2>/dev/null; do
  A=$(tail -1 "$OUT/workers/gpu0.log" 2>/dev/null || true)
  B=$(tail -1 "$OUT/workers/gpu1.log" 2>/dev/null || true)
  echo "[$(date '+%F %T')] GPU0: $A | GPU1: $B" >> "$OUT/run.log"
  sleep 60
done
wait "$P0"; wait "$P1"
"$PY" "$SCRIPT" --phase finalize 2>&1 | tee -a "$OUT/run.log"
"$PY" benchmarks/frozen_sample_embedding_readout/pipeline/build_notebook.py 2>&1 | tee -a "$OUT/run.log"
"$PY" -m jupyter nbconvert --to notebook --execute \
  benchmarks/frozen_sample_embedding_readout/frozen_sample_embedding_readout_benchmark.ipynb \
  --inplace --ExecutePreprocessor.timeout=600 2>&1 | tee -a "$OUT/run.log"

