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

### Colab main sweep (2026-05-21)

Command:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 28 - Hybrid Vocab Scale Sweep/hybrid_vocab_scale_sweep.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Run shape:

- `device=cuda`
- `tokenizer=/content/data_io/trained_tokenizers/bpe/tokenizer.json`
- `vocab_size=65,536`
- `tokens=126,408` (`train=101,126`, `eval=25,282`)
- `context=128x128`, `hidden_size=256`, `numseqs=8`
- `vocab_modes=512`, `hidden_modes=64`, `fourier_mode=64`
- `pom_order=4`, `ordering=token_frequency`, `warmup_steps=1`
- `seeds=1,2,3`

Summary (mean over 3 seeds):

| Variant | Final eval | Params | Peak VRAM | ms/step | Tokens/s | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `spectral-dense-tied` | **4.0146** | 17,523,784 | 1,827.5 MB | 90.80 | 22,577.1 | **best quality**; spectral body + dense tied head |
| `spectral-hybrid-r128-s050` | 6.0116 | 9,266,248 | 1,925.6 MB | 100.15 | 20,483.5 | **best hybrid**; matches Exp 26 anchor scale |
| `dense-tied-attention` | 6.2045 | 18,219,008 | 1,860.5 MB | 87.67 | 23,403.6 | dense body control |
| `spectral-hybrid-r128-s100` | 6.2371 | 9,266,248 | 1,925.6 MB | 100.77 | 20,352.4 | stronger residual did not help |
| `spectral-hybrid-r128-s025` | 6.3599 | 9,266,248 | 1,925.6 MB | 98.57 | 20,802.9 | weaker residual is worse |
| `spectral-hybrid-r128-s200` | 6.6835 | 9,266,248 | 1,925.6 MB | 101.74 | 20,171.2 | unstable starts (~31 eval); scale too hot |
| `spectral-tied-fourier` | 7.0494 | 844,872 | 1,829.2 MB | 90.19 | 22,731.4 | smallest params; worst eval |

Per-seed final eval (eval loss after training):

| Variant | seed=1 | seed=2 | seed=3 |
| --- | ---: | ---: | ---: |
| `spectral-dense-tied` | 3.9400 | 3.9910 | 4.1128 |
| `spectral-hybrid-r128-s050` | 5.9799 | 6.0124 | 6.0425 |
| `spectral-hybrid-r128-s100` | 5.8366 | 6.3431 | 6.5317 |
| `spectral-hybrid-r128-s025` | 6.3343 | 6.3041 | 6.4412 |
| `spectral-hybrid-r128-s200` | 6.6497 | 6.6688 | 6.7321 |
| `dense-tied-attention` | 5.8742 | 6.2043 | 6.5349 |
| `spectral-tied-fourier` | 7.0451 | 7.0139 | 7.0891 |

Plain read:

- **`residual_scale` did not close the gap** to `spectral-dense-tied` (best hybrid ~6.01 vs target ~4.01).
- **`0.50` remains the best hybrid setting** in this sweep; `0.25` is worse, `1.00` is not better on mean eval, and **`2.00` is unstable** (very high initial eval, weak final gain).
- Hybrids save **~49% params** vs dense tied heads (~9.3M vs ~17.5–18.2M) but **do not beat** the spectral-dense-tied quality target at any tested scale.
- **`spectral-tied-fourier` alone** is far too small param-wise and remains the worst eval floor.
- VRAM spread is narrow (~1.83–1.93 GB peak); scale changes quality more than memory here.
- Throughput: dense-tied-attention is fastest; hybrids cost ~12–16% more ms/step with slightly lower tokens/s.

Decision (vs pre-run expectations):

- Do **not** keep tuning rank-128 scale alone — the simple tied low-rank residual shape still caps well above `spectral-dense-tied`.
- **Keep `spectral-dense-tied` as the quality reference** for the spectral body stack.
- Next bridge work should move **beyond scalar scale** (untied / two-sided / different rank), or accept hybrid as a **param/VRAM trade**, not a quality win at this shape.

Tiny local CPU smoke (wiring only):

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-tied-attention` | 11.4741 | 4,333,568 | 118.37 | 540.7 | script wiring check |

The hybrid scale variants need Colab/Linux because the spectral H-level uses FLA GDN.

## Open questions

- Can an **untied** or **two-sided** rank-128 bridge beat ~6.0 eval without jumping back to ~17.5M params?
- Is rank 128 still the right bottleneck, or should rank track `hidden_size` / vocab block structure?
- Does a longer run (more steps / seeds) narrow the ~2.0 eval gap between `spectral-hybrid-r128-s050` and `spectral-dense-tied`?
