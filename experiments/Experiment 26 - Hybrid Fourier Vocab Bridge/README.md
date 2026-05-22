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

### Colab T4 main sweep

Command:

```python
!python "experiments/Experiment 26 - Hybrid Fourier Vocab Bridge/vocab_bridge_sweep.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Run shape:

- `tokens=115,170`
- `train=92,136`
- `eval=23,034`
- `context=128x128`
- `hidden_size=256`
- `numseqs=8`
- `vocab_size=65,536`
- `residual_scale=0.5`

Summary:

| Variant | Final eval | Params | Peak VRAM | ms/step | Tokens/s | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dense-attention` | 2.4711 | 34,996,224 | 2,052.5 MB | 92.63 | 22,151.9 | full-capacity quality anchor |
| `dense-tied-attention` | 5.7842 | 18,219,008 | 1,860.5 MB | 88.76 | 23,108.3 | dense tied control |
| `spectral-tied-fourier` | 6.8271 | 844,872 | 1,829.2 MB | 91.72 | 22,385.9 | pure Fourier is too narrow |
| `spectral-hybrid-r16` | 6.7504 | 1,897,544 | 1,841.3 MB | 96.08 | 21,352.7 | tiny bridge barely helps |
| `spectral-hybrid-r64` | 6.3877 | 5,055,560 | 1,877.4 MB | 98.30 | 20,869.8 | clear but partial improvement |
| `spectral-hybrid-r128` | 5.7886 | 9,266,248 | 1,925.6 MB | 102.27 | 20,080.0 | reaches dense-tied-attention loss with about half params |
| `spectral-dense-tied` | 3.4149 | 17,523,784 | 1,827.5 MB | 91.26 | 22,477.4 | best spectral result |

Plain read:

- The low-rank bridge works in the expected direction: higher rank means better loss.
- Rank 128 is the first useful bridge. It matches `dense-tied-attention` loss while using about half the params.
- It still does not recover the strong `spectral-dense-tied` result, so this simple residual is not enough yet.
- VRAM barely changes across bridge ranks; params change more than peak memory at this tiny scale.
- The next useful knob is probably not more H-level work. It is the vocab bridge shape: rank, scale, maybe separate input/output residual, or learned token basis.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-tied-attention` | 11.5891 | 4,333,568 | 98.78 | 647.9 | script wiring check |

The hybrid head itself is covered by the unit test. The actual hybrid sweep needs Colab/Linux because the default spectral variants use FLA GDN.

Decision:

- Keep the hybrid bridge as a useful middle point.
- Do not claim victory from rank 16 or rank 64.
- Treat rank 128 as the current compressed bridge anchor.
- Keep `spectral-dense-tied` as the quality target for the next vocab/head experiment.

## Open questions

- Does `residual_scale=0.5` need a sweep around rank 128?
- Would rank 192 or 256 keep closing the gap, or is the curve flattening?
- Would an untied or two-sided residual help more than a tied residual?
