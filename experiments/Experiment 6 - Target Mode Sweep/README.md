# Experiment 6 - Target Mode Sweep

## Goal

Run a fairer target comparison by sweeping Fourier mode count for each projection target:

```text
mlp:       48, 64, 96
attention: 48, 64, 96
all:       64, 96, 128
```

Then rerun the top configs with 3 seeds.

## What Changed

- Added `target_mode_sweep.py`.
- Added tests for target-mode parsing and top-k config selection.

## Files Touched

- `experiments/Experiment 6 - Target Mode Sweep/target_mode_sweep.py`
- `experiments/Experiment 6 - Target Mode Sweep/README.md`
- `tests/test_target_mode_sweep.py`

## Default Run

```bash
python "experiments/Experiment 6 - Target Mode Sweep/target_mode_sweep.py" --steps 500 --scan-seed 1 --rerun-seeds 1,2,3 --top-k 3 --device cuda
```

## Current Status

Helper tests and the full two-stage CUDA sweep have run.

Commands:

```bash
python -m pytest tests/test_target_mode_sweep.py -q
python "experiments/Experiment 6 - Target Mode Sweep/target_mode_sweep.py" --steps 500 --scan-seed 1 --rerun-seeds 1,2,3 --top-k 3 --device cuda
```

## One-Seed Scan

CUDA result on RTX 3050 Ti:

```text
dense:
  final_eval=2.9802
  params=289,536

fourier-mlp-48:
  final_eval=3.0066
  params=151,296

fourier-mlp-64:
  final_eval=2.9118
  params=158,464

fourier-mlp-96:
  final_eval=2.9805
  params=178,944

fourier-attention-48:
  final_eval=3.0518
  params=206,592

fourier-attention-64:
  final_eval=2.9522
  params=213,760

fourier-attention-96:
  final_eval=2.9119
  params=234,240

fourier-all-64:
  final_eval=3.0348
  params=82,688

fourier-all-96:
  final_eval=3.0046
  params=123,648

fourier-all-128:
  final_eval=2.9037
  params=142,080
```

Top configs selected for 3-seed rerun:

```text
fourier-all-128
fourier-mlp-64
fourier-attention-96
```

## Three-Seed Rerun

```text
dense:
  runs=3
  mean_final_eval=2.9593
  stdev_final_eval=0.0610
  mean_params=289,536

fourier-all-128:
  runs=3
  mean_final_eval=2.9071
  stdev_final_eval=0.0124
  mean_params=142,080

fourier-mlp-64:
  runs=3
  mean_final_eval=2.9781
  stdev_final_eval=0.0477
  mean_params=158,464

fourier-attention-96:
  runs=3
  mean_final_eval=2.8867
  stdev_final_eval=0.0599
  mean_params=234,240
```

## Read

Best mean eval in this run:

```text
fourier-attention-96
```

Best compression/performance tradeoff:

```text
fourier-all-128
```

This is the first run where `all` works well after giving it enough modes. The earlier `all-64` result was likely too compressed, not proof that all-projection Fourier is bad.

Peak VRAM from the pre-cleanup run is not used for ranking because sequential CUDA allocator state made the later variants look larger than they are. The shared target-sweep runner now clears cache before each variant for future runs.

## Locked Presets

Current locked local presets:

```text
quality preset: config/arch/net/hrm_fourier_attention_quality.yaml
lean preset:    config/arch/net/hrm_fourier_all_lean.yaml
```

Use with Hydra:

```bash
python pretrain.py arch/net@arch=hrm_fourier_attention_quality
python pretrain.py arch/net@arch=hrm_fourier_all_lean
```

Exact locked values:

```yaml
# quality
fourier_linear:
  enabled: true
  target: attention
  in_modes: 96
  out_modes: 96

# lean
fourier_linear:
  enabled: true
  target: all
  in_modes: 128
  out_modes: 128
```
