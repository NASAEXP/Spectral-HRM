# Experiment 20 - DeltaNet H-Level Candidates

## Goal

Skip GLA/GSA and test DeltaNet-family H-level replacements directly:

```text
L-level = PoM
H-level = SLA vs DeltaNet vs preconditioned DeltaNet
context = 48x48
```

This is not the full Triton/FLA implementation yet. It is a small local mixer meant to answer one question first:

```text
Does a DeltaNet-style H-level look alive enough to justify deeper kernel work?
```

## What Changed

- Added `DeltaNetAttention`, a packed PrefixLM-safe delta-rule mixer.
- Added two token mixer modes:
  - `token_mixer="deltanet"`
  - `token_mixer="precond_deltanet"`
- The preconditioned version applies a bounded diagonal scale to the write-side key. This follows the practical hook described by Preconditioned DeltaNet and OSDN, but it is not a full reproduction of either paper's fused kernels.
- Added a 48x48 runner against the current H-level control, `pom-sla`.

## Sources

- [Preconditioned DeltaNet paper](https://arxiv.org/abs/2604.21100)
- [Preconditioned DeltaNet repo](https://github.com/ntumm120/preconditioned-deltanet)
- [OSDN paper](https://arxiv.org/abs/2605.13473)
- [Parallel DeltaNet paper](https://arxiv.org/abs/2406.06484)

## How To Run

```powershell
rtk python "experiments\Experiment 20 - DeltaNet H-Level Candidates\deltanet_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 20 - DeltaNet H-Level Candidates\deltanet_h_level.py" --steps 5 --seeds 1 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `pom-sla` | PoM | SLA | current H control |
| `pom-deltanet` | PoM | DeltaNet | basic online delta-rule H |
| `pom-precond-deltanet` | PoM | preconditioned DeltaNet | diagonal write-key preconditioned H |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 20 - DeltaNet H-Level Candidates\deltanet_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | H-level | Mean Eval Loss | Params | Peak VRAM | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-sla` | SLA | 6.5229 | 135,684 | 329.6 MB | 21.48 | 8,968.1 | current fast H control |
| `pom-deltanet` | DeltaNet | 6.4893 | 140,296 | 332.9 MB | 190.01 | 1,010.9 | best loss, much slower |
| `pom-precond-deltanet` | preconditioned DeltaNet | 6.5989 | 156,808 | 333.3 MB | 194.97 | 985.9 | worse in this small local version |

Read:

- The basic DeltaNet H-level beats SLA on mean loss at `48x48`: `6.4893` vs `6.5229`.
- The price is large: this naive Python recurrence is roughly `9x` slower than SLA.
- The local preconditioned variant does not help yet. It has more parameters, slower speed, and worse loss than both SLA and DeltaNet.
- This does not reject Preconditioned DeltaNet or OSDN as papers. It only says our tiny unfused preconditioner is not enough.
- DeltaNet is worth keeping as a quality candidate if we later use proper FLA-style kernels. For the current local stack, SLA is still the practical H-level default.
