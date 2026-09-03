#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/walt/bridge-rna
PROVENANCE="$ROOT/embeddings/tcga/ours_r7hnr92k_contextual/provenance.json"
RESULTS="$ROOT/benchmarks/tcga_downstream/results"

echo "[orchestrator] waiting for complete contextual-token cache"
while ! "$ROOT/.venv/bin/python" -c "import json; print(json.load(open('$PROVENANCE'))['complete'])" 2>/dev/null | grep -q True; do
  sleep 60
done

echo "[orchestrator] cache complete; launching classification seed 0 across both GPUs"
set +e
CUDA_VISIBLE_DEVICES=0,1 "$ROOT/.venv/bin/python" \
  "$ROOT/benchmarks/tcga_downstream/pipeline/run_attention_pooling.py" \
  --device cuda:0 --data-parallel --batch-size 8 \
  --tasks classification --seeds 0 --heartbeat-seconds 60 \
  > "$RESULTS/attention_classification_seed0.log" 2>&1
CLASSIFICATION_STATUS=$?

echo "[orchestrator] classification complete status=$CLASSIFICATION_STATUS; launching survival across both GPUs"
CUDA_VISIBLE_DEVICES=0,1 "$ROOT/.venv/bin/python" \
  "$ROOT/benchmarks/tcga_downstream/pipeline/run_attention_pooling.py" \
  --device cuda:0 --data-parallel --batch-size 8 \
  --tasks survival --seeds 0 --heartbeat-seconds 60 \
  > "$RESULTS/attention_survival_seed0.log" 2>&1
SURVIVAL_STATUS=$?
set -e
echo "[orchestrator] complete classification_status=$CLASSIFICATION_STATUS survival_status=$SURVIVAL_STATUS"
test "$CLASSIFICATION_STATUS" -eq 0 -a "$SURVIVAL_STATUS" -eq 0
