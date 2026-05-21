# Experiment 2 - Local Holdout Probe

## Goal

Check whether the Fourier MLP signal from Experiment 1 survives a harder local setup:

```text
train text != eval text
multiple seeds
dense vs selected Fourier mode counts
```

This still uses local repository text and byte tokens. It is not Sapient data yet.

## What Changed

- Added `local_holdout_probe.py`.
- Added tests for mode/seed parsing, train/eval token splitting, and result summaries.
- Reused the Experiment 1 local PrefixLM batching and tiny HRM setup.

## Files Touched

- `experiments/Experiment 2 - Local Holdout Probe/local_holdout_probe.py`
- `experiments/Experiment 2 - Local Holdout Probe/README.md`
- `tests/test_local_holdout_probe.py`

## Default Run

```bash
python "experiments/Experiment 2 - Local Holdout Probe/local_holdout_probe.py" --steps 500 --seeds 1,2,3 --modes 48,64,96 --device cuda
```

## What To Track

- Mean holdout eval loss across seeds
- Standard deviation across seeds
- Parameter count
- Peak VRAM
- Whether Fourier keeps up after dense sees a real train/eval split

## Current Status

Helper tests and the default CUDA holdout probe have run.

Commands:

```bash
python -m pytest tests/test_local_holdout_probe.py -q
python "experiments/Experiment 2 - Local Holdout Probe/local_holdout_probe.py" --steps 500 --seeds 1,2,3 --modes 48,64,96 --device cuda
```

CUDA result on RTX 3050 Ti:

```text
tokens=125,088
train=100,070
eval=25,018
steps=500
seeds=1,2,3

dense:
  runs=3
  mean_final_eval=2.9329
  stdev_final_eval=0.0594
  mean_params=289,536
  peak_vram_mb=23.7

fourier-48:
  runs=3
  mean_final_eval=2.8877
  stdev_final_eval=0.0288
  mean_params=151,296
  peak_vram_mb=22.1

fourier-64:
  runs=3
  mean_final_eval=2.8782
  stdev_final_eval=0.0306
  mean_params=158,464
  peak_vram_mb=22.4

fourier-96:
  runs=3
  mean_final_eval=2.9151
  stdev_final_eval=0.0246
  mean_params=178,944
  peak_vram_mb=22.9
```

## Read

On this local byte-text holdout probe, Fourier-48 and Fourier-64 beat dense mean holdout loss while using about 52-55% of the dense parameter count.

This is still a tiny local probe, not Sapient data. The useful takeaway is narrower:

```text
The Fourier MLP idea still looks alive after train/eval separation and multiple seeds.
```
