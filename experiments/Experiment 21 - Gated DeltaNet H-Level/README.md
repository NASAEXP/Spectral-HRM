# Experiment 21 - Gated DeltaNet H-Level

## Goal

Test Gated DeltaNet as the H-level mixer after the plain DeltaNet pass:

```text
L-level = PoM
H-level = SLA vs DeltaNet vs Gated DeltaNet
context = 48x48
```

This keeps the fast/slow split intact. The L-level still uses PoM. Only the slow H-level changes.

## What Changed

- Added `GatedDeltaNetAttention`, a local exact gated-delta update on top of the existing DeltaNet projection path.
- Added `token_mixer="gdn"` so HRM can place GDN only on the H-level via `H_override`.
- Added a 48x48 runner that compares `pom-gdn` against the current `pom-sla` control and the earlier `pom-deltanet` candidate.

## Sources

- [Gated DeltaNet paper](https://arxiv.org/abs/2412.06464)
- [NVLabs GatedDeltaNet repo](https://github.com/NVlabs/GatedDeltaNet)
- [FLA repo](https://github.com/fla-org/flash-linear-attention)

## How To Run

```powershell
rtk python "experiments\Experiment 21 - Gated DeltaNet H-Level\gdn_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 21 - Gated DeltaNet H-Level\gdn_h_level.py" --steps 5 --seeds 1 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `pom-sla` | PoM | SLA | current H control |
| `pom-deltanet` | PoM | DeltaNet | basic online delta-rule H |
| `pom-gdn` | PoM | Gated DeltaNet | gated-delta H |

## FLA Optimized Path

FLA is the right next path if GDN looks worth keeping. The local implementation here is intentionally simple and Python-level, so it measures behavior first, not final speed.

The FLA-backed version should use `fla.layers.GatedDeltaNet` when the package and kernels are available. Experiment 22 tracks that import/kernel gate separately:

```powershell
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py"
```

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 21 - Gated DeltaNet H-Level\gdn_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | H-level | Mean Eval Loss | Params | Peak VRAM | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-sla` | SLA | 6.0505 | 135,684 | 329.6 MB | 23.52 | 8,320.2 | fastest control |
| `pom-deltanet` | DeltaNet | 6.0237 | 140,296 | 332.9 MB | 197.56 | 973.8 | best loss by a hair |
| `pom-gdn` | Gated DeltaNet | 6.0243 | 140,296 | 333.0 MB | 216.08 | 890.0 | basically tied with DeltaNet, slower |

Read:

- GDN is alive as an H-level replacement.
- GDN and DeltaNet are effectively tied on this tiny run: `6.0243` vs `6.0237`.
- Both beat SLA on mean loss here, but SLA is still roughly `8-9x` faster in the local unfused implementation.
- The local GDN code is useful for behavior testing. It is not the final speed story.
- The next serious GDN speed pass needs FLA/Triton on Linux or Colab.
