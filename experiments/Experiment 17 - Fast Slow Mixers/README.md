# Experiment 17 - Fast Slow Mixers

## Goal

Test the slow/fast idea directly:

```text
L-level = fast local mixer
H-level = slow global mixer
```

The first concrete bet is:

```text
L = PoM-style polynomial mixer
H = SPECTRE mixer
```

This keeps the locked Fourier vocab/body path from earlier experiments, but turns checkpointing off so this run also acts as a speed pass.

## What Changed

- Added `PoMAttention`, a PrefixLM-safe polynomial token mixer.
- Added `token_mixer="pom"` to `TransformerConfig`.
- Used the existing `H_override` path to set H and L mixers separately:
  - base `token_mixer` controls L
  - `H_override.token_mixer` controls H
- Added a 2x2 mixer runner:
  - L attention, H attention
  - L attention, H SPECTRE
  - L PoM, H attention
  - L PoM, H SPECTRE

## How To Run

```powershell
rtk python "experiments\Experiment 17 - Fast Slow Mixers\fast_slow_mixers.py" --steps 40 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 17 - Fast Slow Mixers\fast_slow_mixers.py" --steps 5 --seeds 1 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `attention-attention` | attention | attention | dense attention control |
| `attention-spectre` | attention | SPECTRE | slow H only |
| `pom-attention` | PoM | attention | fast L only |
| `pom-spectre` | PoM | SPECTRE | full fast/slow split |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 17 - Fast Slow Mixers\fast_slow_mixers.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | L-level | H-level | Mean Eval Loss | Params | Peak VRAM | Mean Total Time | Train ms/step | Train tok/s | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `attention-attention` | attention | attention | 7.0955 | 131,072 | 243.6 MB | 0.90 s | 18.51 | 2,601.4 | reference |
| `attention-spectre` | attention | SPECTRE | 6.9956 | 168,452 | 244.1 MB | 3.26 s | 74.90 | 643.1 | best SPECTRE-only quality, slower |
| `pom-attention` | PoM | attention | 6.9551 | 135,168 | 243.6 MB | 0.79 s | 17.69 | 2,713.7 | best short-run quality and fastest |
| `pom-spectre` | PoM | SPECTRE | 7.0718 | 172,548 | 244.2 MB | 3.15 s | 73.54 | 653.3 | full split works, not best yet |

Read:

- PoM is the useful fast L-level candidate from this pass.
- H-level SPECTRE works and slightly helps when L stays attention, but it is much slower here.
- The combined `pom-spectre` path is valid, but this short run does not justify locking it.
- `Mean Total Time` includes eval overhead. `Train ms/step` and `Train tok/s` measure only the training loop.
- Next mixer work should tune PoM first, then revisit H-level SPECTRE with a larger context or longer run.
