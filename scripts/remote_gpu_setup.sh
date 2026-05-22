#!/usr/bin/env bash
# One-shot Linux GPU host setup for Spectral-HRM (SSH workflow, no notebooks).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${SPECTRAL_WORK_ROOT:-/root}"
REPO_DIR="${WORK_ROOT}/Spectral-HRM"
DATA_IO_DIR="${WORK_ROOT}/data_io"
TOKENIZER="${DATA_IO_DIR}/trained_tokenizers/bpe/tokenizer.json"

echo "==> work_root=${WORK_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Use a GPU template with NVIDIA drivers." >&2
  exit 1
fi
nvidia-smi || true

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "==> clone Spectral-HRM"
  git clone --depth 1 "${SPECTRAL_REPO_URL:-https://github.com/NASAEXP/Spectral-HRM.git}" "${REPO_DIR}"
fi

if [[ ! -f "${TOKENIZER}" ]]; then
  echo "==> clone data_io (tokenizer)"
  git clone --depth 1 https://github.com/sapientinc/data_io.git "${DATA_IO_DIR}"
fi

cd "${REPO_DIR}"

echo "==> pip deps"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q einops pydantic tokenizers transformers

echo "==> FLA + Triton"
python3 colab/install_fla_colab.py

echo "==> smoke import"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "OK: ${REPO_DIR} ready. Run: bash scripts/run_exp29_followup.sh attention"
