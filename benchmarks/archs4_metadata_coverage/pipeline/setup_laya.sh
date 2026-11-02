#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${BENCHMARK_DIR}/work"
SOURCE_DIR="${WORK_DIR}/laya-src"
VENV_DIR="${WORK_DIR}/laya-gpu-venv"
LAYA_REPOSITORY="https://github.com/NandhaKishorM/laya.git"
LAYA_COMMIT="4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"

mkdir -p "${WORK_DIR}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone "${LAYA_REPOSITORY}" "${SOURCE_DIR}"
fi

git -C "${SOURCE_DIR}" fetch origin "${LAYA_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${LAYA_COMMIT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  torch
"${VENV_DIR}/bin/python" -m pip install --editable "${SOURCE_DIR}"

HF_HOME="${WORK_DIR}/laya-hf" \
  LAYA_DEVICE="cuda" \
  "${VENV_DIR}/bin/python" "${BENCHMARK_DIR}/pipeline/smoke_test_laya.py"
