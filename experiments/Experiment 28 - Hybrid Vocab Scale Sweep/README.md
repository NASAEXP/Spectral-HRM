# Experiment 28 - Hybrid Vocab Scale Sweep

## Goal

Return to the vocab/head path and test whether the rank-128 hybrid bridge was simply using the wrong residual strength.

Experiment 26 showed rank 128 is the first useful low-rank bridge, but it only matched `dense-tied-attention`. This experiment fixes `residual_rank=128` and sweeps `residual_scale`.

## What changed

- Added `hybrid_vocab_scale_sweep.py`.
- Reused Experiment 26's training path.
- Added scale-specific variants for the rank-128 hybrid vocab bridge.

## Variants

| Variant | Body | Vocab/head | Scale | Why it is here |
| --- | --- | --- | ---: | --- |
| `dense-tied-attention` | dense attention | dense tied | 0.0 | dense tied control |
| `spectral-tied-fourier` | PoM + FLA GDN | tied Fourier | 0.0 | pure Fourier floor |
| `spectral-hybrid-r128-s025` | PoM + FLA GDN | Fourier + rank 128 residual | 0.25 | weaker residual |
| `spectral-hybrid-r128-s050` | PoM + FLA GDN | Fourier + rank 128 residual | 0.50 | Experiment 26 anchor |
| `spectral-hybrid-r128-s100` | PoM + FLA GDN | Fourier + rank 128 residual | 1.00 | stronger residual |
| `spectral-hybrid-r128-s200` | PoM + FLA GDN | Fourier + rank 128 residual | 2.00 | aggressive residual |
| `spectral-dense-tied` | PoM + FLA GDN | dense tied | 0.0 | quality target |

## Files touched

- `experiments/Experiment 28 - Hybrid Vocab Scale Sweep/hybrid_vocab_scale_sweep.py`
- `tests/test_hybrid_vocab_scale_sweep.py`
- `experiments/README.md`

## How to run

Fast Colab smoke:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 28 - Hybrid Vocab Scale Sweep/hybrid_vocab_scale_sweep.py" \
  --steps 5 \
  --warmup-steps 1 \
  --seeds 1 \
  --device cuda
```

Main Colab sweep:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 28 - Hybrid Vocab Scale Sweep/hybrid_vocab_scale_sweep.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Local CPU smoke:

```powershell
rtk python "experiments\Experiment 28 - Hybrid Vocab Scale Sweep\hybrid_vocab_scale_sweep.py" --steps 1 --warmup-steps 1 --seeds 1 --variants dense-tied-attention --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16 --eval-batches 1
```

## Verification

Focused tests:

```powershell
rtk python -m pytest tests/test_hybrid_vocab_scale_sweep.py -q
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
| `dense-tied-attention` | 11.4741 | 4,333,568 | 118.37 | 540.7 | script wiring check |

The hybrid scale variants need Colab/Linux because the spectral H-level uses FLA GDN.

Expected decision:

- If scale 1.0 or 2.0 closes the gap to `spectral-dense-tied`, keep the rank-128 bridge and tune scale.
- If all scales remain near Experiment 26, the simple tied low-rank residual shape is the limiter.
- If larger scales destabilize loss, the residual is overpowering the Fourier seed.

## Open questions

- Is `residual_scale=0.5` underpowered?
- Does scale change quality without changing params or much VRAM?
- Should the next bridge be untied/two-sided instead of simply stronger?
