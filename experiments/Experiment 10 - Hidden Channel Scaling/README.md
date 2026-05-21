# Experiment 10 - Hidden Channel Scaling

## Goal

Check whether `fourier-all-tied-fourier-vocab` improves when the hidden channel size is bigger.

This follows the high-mode vocab result from Experiment 9:

```text
512 vocab modes is already maxed locally because local vocab_size=260.
The next meaningful knob is hidden_size and hidden_modes.
```

## What Changed

No model code changed.

This experiment reruns:

```text
fourier-all
fourier-all-tied-fourier-vocab
```

with larger hidden sizes:

```text
hidden_size=96,  fourier_mode=48
hidden_size=128, fourier_mode=64
```

The transformer Fourier mode is scaled with hidden size at roughly half-width.

## Commands

Hidden size 96:

```powershell
rtk python "experiments\Experiment 9 - Vocab Mode Sweep\vocab_mode_sweep.py" --steps 80 --seeds 1,2,3 --device cpu --hidden-size 96 --fourier-mode 48 --mode-pairs 512x64,512x96 --variants fourier-all,fourier-all-tied-fourier-vocab
```

Hidden size 128:

```powershell
rtk python "experiments\Experiment 9 - Vocab Mode Sweep\vocab_mode_sweep.py" --steps 80 --seeds 1,2,3 --device cpu --hidden-size 128 --fourier-mode 64 --mode-pairs 512x64,512x96,512x128 --variants fourier-all,fourier-all-tied-fourier-vocab
```

## Results

| Hidden Size | Variant | Vocab Modes | Mean Eval Loss | Params |
| ---: | --- | ---: | ---: | ---: |
| 96 | fourier-all | control | 3.6884 | 68,352 |
| 96 | fourier-all-tied-fourier-vocab | 512x64 | 3.6945 | 35,072 |
| 96 | fourier-all-tied-fourier-vocab | 512x96 | 3.6957 | 43,392 |
| 128 | fourier-all | control | 3.6843 | 99,328 |
| 128 | fourier-all-tied-fourier-vocab | 512x64 | 3.6896 | 49,408 |
| 128 | fourier-all-tied-fourier-vocab | 512x96 | 3.7180 | 57,728 |
| 128 | fourier-all-tied-fourier-vocab | 512x128 | 3.7850 | 66,048 |

## Read

`fourier-all-tied-fourier-vocab` is much more competitive once hidden size increases.

At hidden size 128:

```text
fourier-all:                  3.6843 eval, 99,328 params
fourier-all + tied vocab:     3.6896 eval, 49,408 params
```

That is nearly the same eval loss with about half the parameters.

Important: scaling hidden modes upward did not help in this local probe. `512x64` beat `512x96` and `512x128`.

## Current Choice

Keep:

```text
hrm_fourier_all_tied_vocab.yaml = 512x64
```

For now, do not scale hidden modes proportionally in the local setup. Scale hidden size first, keep hidden modes at 64, and revisit hidden modes on a larger real vocab/model.

## Open Questions

- Does `512x64` still hold up with more steps?
- Does the combined branch beat `fourier-all` on a larger local text probe?
- At real HRM-Text vocab scale, should hidden modes grow to 128 or 256?
