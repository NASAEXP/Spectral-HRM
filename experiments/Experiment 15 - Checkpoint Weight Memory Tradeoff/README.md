# Experiment 15 - Checkpoint Weight Memory Tradeoff

## Goal

Benchmark and analyze the VRAM and speed trade-offs of the `checkpoint_weight` option in `TiedFourierVocab` under a fake large vocabulary and hidden size scale:

- `vocab_size = 50,000`
- `hidden_size = 1,536`
- `vocab_modes = 512`
- `hidden_modes = 64`
- `batch_size = 4`
- `seq_len = 1,024`

This is a scale-memory test to determine whether we can trade backward-pass compute for lower peak VRAM by freeing the large dense weight matrix `W` and rebuilding it during backward.

At this shape, `W` is roughly:

```text
50,000 x 1,536 x 4 bytes = 307.2 MB
```

## What Changed

- Improved the `checkpoint_weight` implementation in `models/lm_head.py` to jointly checkpoint weight reconstruction and the logits projection.
- Updated `LMHead.forward` to avoid materializing the dense weight upfront when `checkpoint_weight=True`.
- Created a focused benchmark script in `experiments/Experiment 15 - Checkpoint Weight Memory Tradeoff/checkpoint_weight_tradeoff.py`.
- Added correctness tests in `tests/test_checkpoint_weight_tradeoff.py` to verify mathematical equivalence.

## How To Run

```powershell
rtk python "experiments\Experiment 15 - Checkpoint Weight Memory Tradeoff\checkpoint_weight_tradeoff.py"
```

## Results

Local NVIDIA GeForce RTX 3050 Ti Laptop GPU result:

| Condition | Peak VRAM | Forward Time | Backward Time | Max Gradient Difference |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint OFF | 2,778.17 MB | 199.12 ms | 307.32 ms | - |
| Checkpoint ON | 2,485.32 MB | 217.29 ms | 336.62 ms | `7.45e-09` |

## Key Metrics

- VRAM savings: **292.84 MB** or **10.5%**
- Forward overhead: **+18.17 ms** or **+9.1%**
- Backward overhead: **+29.30 ms** or **+9.5%**
- Net overhead: **+47.47 ms** per train step
- Gradient accuracy: max coefficient-gradient difference is `7.45e-09`

## Findings

The previous `checkpoint_weight` implementation checkpointed only reconstruction of `W`. That was not enough for logits, because `W` was still passed into `F.linear` outside the checkpoint and PyTorch had to keep it for backward.

The improved path jointly checkpoints:

```text
coefficients -> reconstruct W -> F.linear(hidden_states, W)
```

That keeps `W` local to the checkpointed block. It is computed in forward, freed, and reconstructed during backward. In this benchmark it reclaims about **293 MB** of VRAM with about a **9%** compute slowdown.

## Locked Decision

Carry forward for 4GB/survival runs:

```text
checkpoint_weight = true
control           = checkpoint_weight false
```

Preset:

```text
config/arch/net/hrm_fourier_all_tied_vocab_checkpoint.yaml
```

This is locked as the low-VRAM preset, not as the speed preset. Keep `hrm_fourier_all_tied_vocab.yaml` as the faster non-checkpoint control.
