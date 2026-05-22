#!/usr/bin/env bash
# Run on the Vast instance after uploading vast_pilot_data.tar.gz
set -euo pipefail

WORK="${WORK:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORK/Spectral-HRM}"
DATA_TAR="${DATA_TAR:-$WORK/data_upload.tar}"
DATA_DIR="${DATA_DIR:-$WORK/data/sampled}"

mkdir -p "$WORK/data"
if [[ -f "$DATA_TAR" ]]; then
  mkdir -p "$DATA_DIR"
  tar -xzf "$DATA_TAR" -C "$DATA_DIR"
  echo "data unpacked to $DATA_DIR"
  ls -la "$DATA_DIR"
else
  echo "WARN: missing $DATA_TAR — upload vast_pilot_data.tar.gz first"
fi

if [[ -d "$REPO_DIR" ]]; then
  cd "$REPO_DIR"
  pip install -r requirements.txt
  if [[ -f requirements-fla.txt ]]; then
    pip install -r requirements-fla.txt
  fi
  python scripts/vast_pilot/verify_dataset.py "$DATA_DIR" --batch-size 8192 --peek-batches 1
  echo "bootstrap ok: repo=$REPO_DIR data=$DATA_DIR"
else
  echo "WARN: clone repo to $REPO_DIR then re-run pip + verify"
fi
