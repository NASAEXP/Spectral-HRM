# Experiment 27 - TRM Island PolyAttn

## Goal

Test the idea of keeping the current best spectral body, but adding a sparse TRM-style recursive island inside the H-level only.

This does not replace PoM or FLA GDN globally:

- L-level stays `PoM`.
- H-level stays `FLA GDN`.
- Every 4th H-level layer becomes a small recursive island.
- Only that island uses `PolyAttention`.

## What changed

- Added `PolyAttention`, a small order-3 polynomial attention island based on the `h(x1, x2, x3) = x1*x2 + x2*x3` pattern.
- Added `TRMIslandBlock`, which recursively refines a local state for `trm_island_steps`.
- Added Transformer config knobs:
  - `trm_island_every`
  - `trm_island_mixer`
  - `trm_island_steps`
- Added `trm_island_polyattn.py`.

## Variants

| Variant | Body | Island | Why it is here |
| --- | --- | --- | --- |
| `spectral-dense-tied-deep` | PoM L + FLA GDN H | none | deeper current-best anchor |
| `spectral-trm-island-attention` | PoM L + FLA GDN H | recursive attention | separates recursion effect from PolyAttn effect |
| `spectral-trm-island-polyattn` | PoM L + FLA GDN H | recursive PolyAttn | user idea under test |

The default uses `n_layers=8` with `half_layers=True`, so H and L each get four layers. `trm_island_every=4` means exactly one H-level island in this small run.

## Files touched

- `models/layers.py`
- `models/transformer.py`
- `experiments/Experiment 27 - TRM Island PolyAttn/trm_island_polyattn.py`
- `tests/test_polyattn_trm_island.py`
- `tests/test_trm_island_polyattn_experiment.py`
- `experiments/README.md`

## How to run

Fast Colab smoke:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 27 - TRM Island PolyAttn/trm_island_polyattn.py" \
  --steps 5 \
  --warmup-steps 1 \
  --seeds 1 \
  --device cuda
```

Main Colab comparison:

```python
%cd /content/Spectral-HRM
!git pull
!python "experiments/Experiment 27 - TRM Island PolyAttn/trm_island_polyattn.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Local CPU smoke:

```powershell
rtk python "experiments\Experiment 27 - TRM Island PolyAttn\trm_island_polyattn.py" --steps 1 --warmup-steps 1 --seeds 1 --variants dense-tied-attention --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16 --eval-batches 1 --n-layers 8
```

## Verification

Focused tests:

```powershell
rtk python -m pytest tests/test_polyattn_trm_island.py tests/test_trm_island_polyattn_experiment.py -q
```

Broader checks:

```powershell
rtk python -m pytest -q
rtk python -m compileall experiments tests colab models
rtk git diff --check
```

## Current read

### Colab T4 main comparison

Command:

```python
!python "experiments/Experiment 27 - TRM Island PolyAttn/trm_island_polyattn.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Run shape:

- `tokens=126,408`
- `train=101,126`
- `eval=25,282`
- `context=128x128`
- `hidden_size=256`
- `numseqs=8`
- `n_layers=8`
- `trm_island_every=4`
- `trm_island_steps=2`

Summary:

| Variant | Final eval | Params | Peak VRAM | ms/step | Tokens/s | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `spectral-dense-tied-deep` | 3.1967 | 19,763,488 | 2,055.2 MB | 135.68 | 15,106.7 | best result; no island |
| `spectral-trm-island-attention` | 3.4288 | 19,758,296 | 2,103.9 MB | 144.90 | 14,136.1 | recursion island hurts |
| `spectral-trm-island-polyattn` | 3.4544 | 19,889,368 | 2,134.5 MB | 159.29 | 12,859.7 | PolyAttn island is slower and slightly worse |

Plain read:

- The TRM island did not help in this setup.
- Attention island was worse than the no-island baseline.
- PolyAttn island was also worse, plus slower and higher VRAM.
- This does not kill PolyAttn forever, but it says not to mix it into the current best body right now.
- The practical path should return to vocab/head work or longer-run validation of the current best stack.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-tied-attention` | 11.4838 | 4,751,360 | 117.20 | 546.1 | script wiring check |

The actual island variants need Colab/Linux because the default H-level uses FLA GDN.

Decision:

- Do not lock TRM islands into the current best stack.
- Do not continue PolyAttn island work until we have a stronger reason.
- Keep this code as a branchable experiment, but treat `spectral-dense-tied-deep` as the body anchor.
- Go back to vocab/head work or run a longer validation of the current best body.

## Open questions

- Would islands help only on deeper/larger models?
- Would PolyAttn help only on tasks with stronger compositional structure?
- Is replacing one FLA GDN block too disruptive for HRM-style recurrence?
