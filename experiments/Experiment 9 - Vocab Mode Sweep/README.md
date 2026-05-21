# Experiment 9 - Vocab Mode Sweep

## Goal

Run the same four Experiment 8 variants while changing only the tied Fourier vocab matrix knobs.

The knobs are:

```text
vocab_modes x hidden_modes
```

Higher modes mean a larger coefficient grid, more parameters, and usually less pressure on quality.

## Variants

```text
dense
tied-fourier-vocab
fourier-all
fourier-all-tied-fourier-vocab
```

Dense and `fourier-all` do not use the vocab knobs, but they are repeated as controls so each sweep block has the same four rows.

## How To Run

```powershell
rtk python "experiments\Experiment 9 - Vocab Mode Sweep\vocab_mode_sweep.py" --steps 80 --seeds 1,2,3 --device cpu --mode-pairs 80x32,128x64,160x64,224x64
```

## Result

Command:

```powershell
rtk python "experiments\Experiment 9 - Vocab Mode Sweep\vocab_mode_sweep.py" --steps 80 --seeds 1,2,3 --device cpu --mode-pairs 80x32,128x64,160x64,224x64
```

Summary:

| Variant | Modes | Mean Eval Loss | Params |
| --- | ---: | ---: | ---: |
| dense | control | 3.6445 | 172,544 |
| fourier-all | control | 3.6014 | 41,472 |
| tied-fourier-vocab | 80x32 | 4.0423 | 141,824 |
| tied-fourier-vocab | 128x64 | 3.6979 | 147,456 |
| tied-fourier-vocab | 160x64 | 3.6291 | 149,504 |
| tied-fourier-vocab | 224x64 | 3.6536 | 153,600 |
| fourier-all-tied-fourier-vocab | 80x32 | 4.0959 | 10,752 |
| fourier-all-tied-fourier-vocab | 128x64 | 3.7552 | 16,384 |
| fourier-all-tied-fourier-vocab | 160x64 | 3.7468 | 18,432 |
| fourier-all-tied-fourier-vocab | 224x64 | 3.7412 | 22,528 |

Read:

```text
80x32 is too tight.
160x64 is the best tied-vocab-only setting in this sweep.
224x64 is the best combined fourier-all + tied-vocab setting, but it is still behind fourier-all alone.
```

Initial preset choice after this sweep:

```text
hrm_tied_fourier_vocab.yaml = 160x64
hrm_fourier_all_tied_vocab.yaml = 224x64
```

## High-Mode Follow-Up

Command:

```powershell
rtk python "experiments\Experiment 9 - Vocab Mode Sweep\vocab_mode_sweep.py" --steps 80 --seeds 1,2,3 --device cpu --mode-pairs 224x64,256x64,512x64 --variants fourier-all,fourier-all-tied-fourier-vocab
```

Summary:

| Variant | Modes | Mean Eval Loss | Params |
| --- | ---: | ---: | ---: |
| fourier-all | control | 3.6014 | 41,472 |
| fourier-all-tied-fourier-vocab | 224x64 | 3.7412 | 22,528 |
| fourier-all-tied-fourier-vocab | 256x64 | 3.7386 | 24,576 |
| fourier-all-tied-fourier-vocab | 512x64 | 3.7175 | 24,832 |

Read:

```text
Increasing vocab modes helps the combined variant.
512x64 is still behind fourier-all, but it is the best combined setting so far.
On the local toy vocab, 512 vocab modes clamps to vocab_size=260, so this is basically the max-vocab-side version.
```

Updated combined preset:

```text
hrm_fourier_all_tied_vocab.yaml = 512x64
```
