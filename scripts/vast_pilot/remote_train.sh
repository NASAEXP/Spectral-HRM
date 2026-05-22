#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Spectral-HRM
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"

LOG=/workspace/train.log
exec > >(tee -a "$LOG") 2>&1
echo "=== train start $(date -Iseconds) ==="

bash scripts/vast_pilot/smoke_pretrain_vast.sh /workspace/data/sampled 8192

python3 pretrain.py --config-name cfg_pretrain_vast \
  data.path=/workspace/data/sampled \
  global_batch_size=8192 \
  epochs=1

echo "=== train done $(date -Iseconds) ==="
ls -la checkpoints/Spectral-HRM-Vast-Pilot/ || true
