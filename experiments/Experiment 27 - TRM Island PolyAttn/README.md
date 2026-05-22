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

No Colab results yet.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-tied-attention` | 11.4838 | 4,751,360 | 117.20 | 546.1 | script wiring check |

The actual island variants need Colab/Linux because the default H-level uses FLA GDN.

Expected decision:

- If `spectral-trm-island-attention` helps, the recursive island itself is useful.
- If `spectral-trm-island-polyattn` beats the attention island, PolyAttn is worth keeping in the island.
- If both islands lose badly, return to vocab/head work before adding more body complexity.

## Open questions

- Is one island every 4 H layers enough?
- Is `trm_island_steps=2` enough?
- Does PolyAttn help only on longer runs where composition matters more?
