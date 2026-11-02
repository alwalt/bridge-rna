#!/usr/bin/env bash
set -euo pipefail
PYTHON=${PYTHON:-/home/walt/miniconda3/envs/biofm/bin/python}
"$PYTHON" "$(dirname "$0")/run_benchmark.py"
"$PYTHON" "$(dirname "$0")/expression_baseline.py"
"$PYTHON" "$(dirname "$0")/summarize.py"
"$PYTHON" "$(dirname "$0")/test_invariants.py"
"$PYTHON" "$(dirname "$0")/finalize.py"
