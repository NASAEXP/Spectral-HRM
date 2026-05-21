# Experiment 16 - Locked Survival Stack

## Goal

Check whether the locked Spectral-HRM pieces still work together:

```text
Fourier-all body
+ tied Fourier vocab
+ Sapient BPE vocab
+ token_frequency vocab ordering
+ checkpoint_weight=true
```

This is an integration experiment. It tests the locked stack against controls instead of adding a new trick.

## What Changed

- Added `fourier-all-tied-fourier-vocab-bias-checkpoint` to the Experiment 8 probe variants.
- Added a 2x2 runner:
  - BPE-rank order, checkpoint off
  - token-frequency order, checkpoint off
  - BPE-rank order, checkpoint on
  - token-frequency order, checkpoint on

## How To Run

```powershell
rtk python "experiments\Experiment 16 - Locked Survival Stack\locked_survival_stack.py" --steps 80 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 16 - Locked Survival Stack\locked_survival_stack.py" --steps 10 --seeds 1 --device cuda
```

## Variants

| Variant | Ordering | Checkpoint | Meaning |
| --- | --- | --- | --- |
| `control-bpe-rank` | `bpe_rank` | off | reference control |
| `frequency-order` | `token_frequency` | off | ordering-only gain |
| `checkpoint-control` | `bpe_rank` | on | checkpoint-only cost/gain |
| `survival-locked` | `token_frequency` | on | locked low-VRAM stack |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 16 - Locked Survival Stack\locked_survival_stack.py" --steps 80 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Variant | Mean Eval Loss | Params | Peak VRAM | Mean Time | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `control-bpe-rank` | 10.0147 | 131,072 | 243.6 MB | 1.49 s | reference control |
| `frequency-order` | 7.0827 | 131,072 | 243.6 MB | 1.48 s | ordering win |
| `checkpoint-control` | 10.0147 | 131,072 | 243.7 MB | 1.86 s | same loss, slower |
| `survival-locked` | 7.0827 | 131,072 | 243.7 MB | 1.86 s | locked stack works |

Read:

- `token_frequency` is the quality gain.
- `checkpoint_weight=true` preserves the exact same loss path in this local run.
- At `hidden_size=128`, the dense generated vocab matrix is too small for visible VRAM savings, so Experiment 15 remains the scale-memory proof.
- The locked low-VRAM stack is valid, but use it when VRAM matters. For speed-only local probes, keep checkpointing off.
