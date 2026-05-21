# Experiment 4 - Tiny Sapient Format

## Goal

Move from custom local byte-text batches to the actual HRM-Text `V1Dataset` file shape:

```text
metadata.json
tokens.npy
epoch_0/inst_start.npy
epoch_0/inst_len.npy
epoch_0/resp_start.npy
epoch_0/resp_len.npy
```

This is still a generated tiny dataset, not Sapient's released corpus.

## What Changed

- Added `tiny_sapient_probe.py`.
- Added tests that build a tiny dataset and load it through `V1Dataset`.

## Files Touched

- `experiments/Experiment 4 - Tiny Sapient Format/tiny_sapient_probe.py`
- `experiments/Experiment 4 - Tiny Sapient Format/README.md`
- `tests/test_tiny_sapient_format.py`

## Default Run

```bash
python "experiments/Experiment 4 - Tiny Sapient Format/tiny_sapient_probe.py" --steps 120 --device cuda
```

## Current Status

Tests and a CUDA smoke run have completed.

Commands:

```bash
python -m pytest tests/test_tiny_sapient_format.py -q
python "experiments/Experiment 4 - Tiny Sapient Format/tiny_sapient_probe.py" --steps 120 --device cuda
```

CUDA result on RTX 3050 Ti:

```text
batches=9
vocab_size=512
max_seq_len=23
steps=120

dense:
  loss 6.7595 -> 0.0230
  params=337,920
  peak_vram_mb=26.2

fourier-64:
  loss 6.7135 -> 0.0509
  params=206,848
  peak_vram_mb=26.5
```

## Read

This experiment proves the tiny local HRM-Text `V1Dataset` path runs end to end without FlashAttention or Numba installed.

Dense memorized the generated shard faster than Fourier-64. That is expected for a tiny synthetic dataset and should not be read as a general quality result.

The useful result is narrower:

```text
The Fourier model can train through the actual HRM-Text dataset format locally.
```
