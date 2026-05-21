# Experiment 5 - Fourier Attention Targets

## Goal

Find where Fourier projections belong inside the HRM-Text block:

```text
dense
Fourier MLP only
Fourier attention only
Fourier MLP + attention
```

This changes only projection weight parameterization. It does not change the attention algorithm.

## What Changed

- Added `fourier_target_sweep.py`.
- Added tests for attention-only and all-projection config wiring.
- Added tests for target parsing and variant naming.

## Files Touched

- `experiments/Experiment 5 - Fourier Attention Targets/fourier_target_sweep.py`
- `experiments/Experiment 5 - Fourier Attention Targets/README.md`
- `tests/test_fourier_linear.py`
- `tests/test_fourier_target_sweep.py`

## Default Run

```bash
python "experiments/Experiment 5 - Fourier Attention Targets/fourier_target_sweep.py" --steps 500 --seeds 1,2,3 --targets dense,mlp,attention,all --mode 64 --device cuda
```

## Current Status

Target wiring tests and the default CUDA sweep have run.

Commands:

```bash
python -m pytest tests/test_fourier_target_sweep.py tests/test_fourier_linear.py -q
python "experiments/Experiment 5 - Fourier Attention Targets/fourier_target_sweep.py" --steps 500 --seeds 1,2,3 --targets dense,mlp,attention,all --mode 64 --device cuda
```

CUDA result on RTX 3050 Ti:

```text
steps=500
seeds=1,2,3
mode=64

dense:
  runs=3
  mean_final_eval=2.9593
  stdev_final_eval=0.0610
  mean_params=289,536

fourier-mlp-64:
  runs=3
  mean_final_eval=2.9781
  stdev_final_eval=0.0477
  mean_params=158,464

fourier-attention-64:
  runs=3
  mean_final_eval=2.9538
  stdev_final_eval=0.0107
  mean_params=213,760

fourier-all-64:
  runs=3
  mean_final_eval=3.0391
  stdev_final_eval=0.0062
  mean_params=82,688
```

## Read

In this local target sweep:

```text
Fourier attention-only slightly beat dense mean eval.
Fourier MLP-only was close but slightly behind dense in this run.
Fourier all-projections at 64 modes was too compressed.
```

This does not replace Experiment 2, where MLP-only won. It means target choice is sensitive at this tiny scale and deserves a combined mode sweep before changing the default preset.
