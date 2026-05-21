# Experiment 18 - H-Level SLA

## Goal

Test **Softmax Linear Attention (SLA)** as an H-level replacement while keeping the current fast side fixed:

```text
L-level = PoM
H-level = attention vs SLA vs SPECTRE
```

SLA is interesting here because it keeps a linear attention backbone but adds softmax competition across heads. That matches the H-level job: keep global selection sharp without returning to full token-token attention.

Paper: [Softmax Linear Attention: Reclaiming Global Competition](https://arxiv.org/abs/2602.01744)

## What Changed

- Added `SLAAttention`, a packed PrefixLM-safe SLA-style mixer.
- Added `token_mixer="sla"` to `TransformerConfig`.
- Added a focused H-level runner with L locked to PoM:
  - H attention
  - H SLA
  - H SPECTRE

## How To Run

```powershell
rtk python "experiments\Experiment 18 - H-Level SLA\h_level_sla.py" --steps 40 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 18 - H-Level SLA\h_level_sla.py" --steps 5 --seeds 1 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `pom-attention` | PoM | attention | current best H control |
| `pom-sla` | PoM | SLA | new H candidate |
| `pom-spectre` | PoM | SPECTRE | prior spectral H control |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 18 - H-Level SLA\h_level_sla.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | L-level | H-level | Mean Eval Loss | Params | Peak VRAM | Mean Total Time | Train ms/step | Train tok/s | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-attention` | PoM | attention | 6.8123 | 135,168 | 243.6 MB | 0.95 s | 18.85 | 2,548.6 | current H control |
| `pom-sla` | PoM | SLA | 6.7913 | 135,684 | 243.6 MB | 0.89 s | 20.09 | 2,391.2 | best loss, slight speed cost |
| `pom-spectre` | PoM | SPECTRE | 6.8983 | 172,548 | 244.2 MB | 3.32 s | 76.47 | 628.2 | slower and worse here |

Read:

- SLA beats the attention H-control on mean loss in this run: `6.7913` vs `6.8123`.
- SLA has almost the same parameter count and peak VRAM as attention.
- SLA is slower inside the training loop: `20.09 ms/step` vs `18.85 ms/step`.
- SPECTRE is not competitive in this short local setup.
- SLA is worth keeping as the next H-level candidate, but it needs a longer-context pass before locking.
