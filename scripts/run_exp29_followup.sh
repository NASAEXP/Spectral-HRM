#!/usr/bin/env bash
# Exp 29 follow-up sweeps via shell (no notebook). Usage: run_exp29_followup.sh [attention|fla|all]
set -euo pipefail

MODE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WORK_ROOT="${SPECTRAL_WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
TOKENIZER="${SPECTRAL_TOKENIZER:-${WORK_ROOT}/data_io/trained_tokenizers/bpe/tokenizer.json}"
RUN_DIR="${REPO_ROOT}/runs/exp29"
mkdir -p "${RUN_DIR}"

STEPS="${STEPS:-40}"
SEEDS="${SEEDS:-1,2,3}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -f "${TOKENIZER}" ]]; then
  echo "ERROR: tokenizer not found: ${TOKENIZER}" >&2
  echo "Run scripts/remote_gpu_setup.sh first." >&2
  exit 1
fi

run_attention() {
  echo "==> Phase A: hidden_modes (PoM L + attention H)"
  python3 "experiments/Experiment 29 - Vocab Head Pareto Sweep/colab_followup_sweep.py" \
    --steps "${STEPS}" --warmup-steps 1 --seeds "${SEEDS}" --device "${DEVICE}" \
    --tokenizer-path "${TOKENIZER}" \
    --variants pom-tied-fourier,pom-multiscale-fourier,pom-tiered-hot-fourier \
    --hidden-modes-list 64,128,192,256 \
    2>&1 | tee "${RUN_DIR}/hidden_modes_$(date +%Y%m%d_%H%M%S).log"
}

run_fla() {
  echo "==> Phase B: FLA-GDN port"
  python3 "experiments/Experiment 22 - FLA GDN Kernel Probe/fla_gdn_probe.py" --require-ready
  HIDDEN_MODES="${HIDDEN_MODES:-64}"
  if [[ -z "${MULTISCALE:-}" ]]; then
    second_h=$(( HIDDEN_MODES * 2 ))
    (( second_h > 256 )) && second_h=256
    MULTISCALE="512,${HIDDEN_MODES};256,${second_h}"
  fi
  python3 "experiments/Experiment 29 - Vocab Head Pareto Sweep/fla_gdn_port_sweep.py" \
    --steps "${STEPS}" --warmup-steps 1 --seeds "${SEEDS}" --device "${DEVICE}" \
    --tokenizer-path "${TOKENIZER}" \
    --hidden-modes "${HIDDEN_MODES}" --multiscale-specs "${MULTISCALE}" \
    2>&1 | tee "${RUN_DIR}/fla_port_$(date +%Y%m%d_%H%M%S).log"
}

case "${MODE}" in
  attention) run_attention ;;
  fla) run_fla ;;
  all)
    run_attention
    run_fla
    ;;
  *)
    echo "Usage: $0 [attention|fla|all]" >&2
    exit 1
    ;;
esac

echo "Done. Logs in ${RUN_DIR}"
