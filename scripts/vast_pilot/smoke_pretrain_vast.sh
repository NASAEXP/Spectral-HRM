#!/usr/bin/env bash
# Quick OOM / import smoke on a Vast instance. Not a full train.
set -euo pipefail

DATA_PATH="${1:-/workspace/data/sampled}"
BATCH="${2:-4096}"

cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1

python pretrain.py --config-name cfg_pretrain_vast \
  data.path="${DATA_PATH}" \
  global_batch_size="${BATCH}" \
  epochs=1 \
  log_interval=1 \
  lr_warmup_steps=1

echo "smoke ok: data=${DATA_PATH} global_batch_size=${BATCH}"
