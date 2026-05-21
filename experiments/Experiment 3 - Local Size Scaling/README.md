# Experiment 3 - Local Size Scaling

## Goal

Check whether the Fourier-64 signal survives as the tiny local HRM hidden size grows.

This is still local repo byte-text, not Sapient data.

## What Changed

- Added `local_size_scaling.py`.
- Added helper tests for size parsing and summary formatting.
- Reused Experiment 2 train/eval holdout code.

## Files Touched

- `experiments/Experiment 3 - Local Size Scaling/local_size_scaling.py`
- `experiments/Experiment 3 - Local Size Scaling/README.md`
- `tests/test_local_size_scaling.py`

## Default Run

```bash
python "experiments/Experiment 3 - Local Size Scaling/local_size_scaling.py" --steps 500 --seed 1 --hidden-sizes 96,128,192,256 --mode 64 --device cuda
```

## Current Status

Helper tests and the one-seed CUDA size sweep have run.

Commands:

```bash
python -m pytest tests/test_local_size_scaling.py -q
python "experiments/Experiment 3 - Local Size Scaling/local_size_scaling.py" --steps 500 --seed 1 --hidden-sizes 96,128,192,256 --mode 64 --device cuda
```

CUDA result on RTX 3050 Ti:

```text
hidden=96:
  dense_eval=3.0169
  fourier_eval=2.8353
  delta=-0.1816
  param_ratio=54.7%

hidden=128:
  dense_eval=2.8868
  fourier_eval=2.9301
  delta=0.0434
  param_ratio=57.8%

hidden=192:
  dense_eval=2.9992
  fourier_eval=2.8505
  delta=-0.1486
  param_ratio=63.5%

hidden=256:
  dense_eval=2.9623
  fourier_eval=3.0104
  delta=0.0481
  param_ratio=51.1%
```

## Read

Fourier-64 remains competitive as hidden size grows, but the result is not monotonic from one seed:

```text
wins: hidden 96, 192
slight losses: hidden 128, 256
```

This is enough to justify moving to a tiny HRM-Text-format data path, but not enough to claim size scaling is solved.
