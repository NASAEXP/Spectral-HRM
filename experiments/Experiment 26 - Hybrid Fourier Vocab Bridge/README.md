# Experiment 26 - Hybrid Fourier Vocab Bridge

## Goal

Find a middle path between pure tied Fourier vocab and full dense tied vocab.

Experiment 25 said the spectral body is good, but pure Fourier vocab/head is too narrow. This experiment keeps the Fourier seed and adds a small low-rank residual:

```text
W = W_fourier + residual_scale * (token_residual @ hidden_residual)
```

The same generated `W` is used for both input embedding and LM head, so weight tying stays intact.

## What changed

- Added `HybridFourierLowRankVocab` in `models/lm_head.py`.
- Added `vocab_head.type = hybrid_fourier_lowrank`.
- Added `residual_rank` and `residual_scale` vocab-head config knobs.
- Added `vocab_bridge_sweep.py`.

## Variants

| Variant | Body | Vocab/head | Why it is here |
| --- | --- | --- | --- |
| `dense-attention` | dense attention | untied dense | full-capacity quality anchor |
| `dense-tied-attention` | dense attention | dense tied | fair tied-vocab dense control |
| `spectral-tied-fourier` | PoM + FLA GDN | tied Fourier | lower bound for pure compression |
| `spectral-hybrid-r16` | PoM + FLA GDN | Fourier + rank 16 residual | tiny bridge |
| `spectral-hybrid-r64` | PoM + FLA GDN | Fourier + rank 64 residual | medium bridge |
| `spectral-hybrid-r128` | PoM + FLA GDN | Fourier + rank 128 residual | large bridge |
| `spectral-dense-tied` | PoM + FLA GDN | dense tied | best current spectral anchor |

## Files touched

- `models/lm_head.py`
- `experiments/Experiment 26 - Hybrid Fourier Vocab Bridge/vocab_bridge_sweep.py`
- `tests/test_tied_fourier_vocab.py`
- `tests/test_vocab_bridge_sweep.py`
- `experiments/README.md`

## How to run

Fast Colab smoke:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 26 - Hybrid Fourier Vocab Bridge/vocab_bridge_sweep.py" \
  --steps 5 \
  --warmup-steps 1 \
  --seeds 1 \
  --device cuda
```

Main Colab sweep:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 26 - Hybrid Fourier Vocab Bridge/vocab_bridge_sweep.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Local CPU smoke:

```powershell
rtk python "experiments\Experiment 26 - Hybrid Fourier Vocab Bridge\vocab_bridge_sweep.py" --steps 1 --warmup-steps 1 --seeds 1 --variants dense-tied-attention --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16 --eval-batches 1
```

## Verification

Focused tests:

```powershell
rtk python -m pytest tests/test_tied_fourier_vocab.py::test_lm_head_can_use_hybrid_fourier_lowrank_vocab tests/test_vocab_bridge_sweep.py -q
```

Broader checks:

```powershell
rtk python -m pytest -q
rtk python -m compileall experiments tests colab models
rtk git diff --check
```

## Current read

No Colab results yet.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-tied-attention` | 11.5891 | 4,333,568 | 98.78 | 647.9 | script wiring check |

The hybrid head itself is covered by the unit test. The actual hybrid sweep needs Colab/Linux because the default spectral variants use FLA GDN.

Expected decision:

- If rank 16 or 64 closes much of the pure Fourier loss gap, keep the hybrid bridge.
- If only rank 128 works, the bridge may still be useful but less dramatic.
- If all hybrid ranks stay close to pure Fourier, the issue is not just missing token-specific capacity.

## Open questions

- Which rank gives the best loss-per-parameter tradeoff?
- Does `residual_scale=0.5` need a sweep?
- Does the hybrid bridge stay useful on longer runs, or only on tiny-data fitting?
