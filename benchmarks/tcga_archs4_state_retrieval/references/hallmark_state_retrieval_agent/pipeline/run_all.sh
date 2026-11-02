#!/usr/bin/env bash
set -euo pipefail
PY=/home/walt/bridge-rna/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p results
exec > >(tee -a results/run.log) 2>&1
echo "Expected runtime: minutes; Bridge inference is skipped because compatible frozen embeddings exist."
$PY pipeline/audit_and_build.py
$PY pipeline/run_queries.py
$PY pipeline/make_report.py

