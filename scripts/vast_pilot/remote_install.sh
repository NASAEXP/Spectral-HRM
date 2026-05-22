#!/usr/bin/env bash
set -euo pipefail
LOG=/workspace/install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== install start $(date -Iseconds) ==="

export PIP_NO_CACHE_DIR=1
export MAX_JOBS="${MAX_JOBS:-8}"

pip install -U pip wheel setuptools packaging ninja

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install packaging ninja wheel setuptools setuptools-scm
MAX_JOBS="$MAX_JOBS" NVCC_THREADS=2 pip install --no-build-isolation \
  "flash_attn_3 @ git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper"

cd /workspace/Spectral-HRM
pip install flash-linear-attention==0.5.0
pip install coolname datasets einops hydra-core numba numpy omegaconf pydantic PyYAML safetensors sympy tqdm transformers wandb

python3 -c "import flash_attn_interface; import torch; print('flash ok')"
python3 scripts/vast_pilot/verify_dataset.py /workspace/data/sampled --batch-size 8192 --peek-batches 1

echo "=== install done $(date -Iseconds) ==="
