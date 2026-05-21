# Experiment 23 - Free Colab FLA Probe

## Goal

Use free Google Colab as the first Linux/CUDA gate for FLA.

This is not the serious training run. It answers two smaller questions:

```text
1. Does free Colab give us a GPU runtime right now?
2. Can that runtime import FLA + Triton + fla.layers.GatedDeltaNet?
```

## Why Free Colab First

Local Windows cannot import the optimized FLA GDN layer because Triton is unavailable here. Colab gives us a Linux runtime without re-enabling Hyper-V/WSL on the laptop.

Google's own Colab FAQ says free resources include GPUs, but they are not guaranteed and can fluctuate. So this experiment is only a smoke gate.

## Files

- `colab/free_fla_gdn_probe.ipynb`
- `experiments/Experiment 22 - FLA GDN Kernel Probe/fla_gdn_probe.py`
- `experiments/Experiment 21 - Gated DeltaNet H-Level/gdn_h_level.py`
- `requirements-fla.txt`

## How To Run

Open the notebook:

```text
colab/free_fla_gdn_probe.ipynb
```

In Colab:

```text
Runtime > Change runtime type > GPU
Run all
```

The notebook will:

- Print `nvidia-smi`.
- Clone `NASAEXP/Spectral-HRM`.
- Clone `sapientinc/data_io` for the tokenizer file.
- Install the optional FLA dependency.
- Run Experiment 22's FLA import gate.
- Run a tiny Experiment 21 GDN smoke on CUDA.

## Success

The FLA gate should print:

```text
status=ready
```

Then we can add the actual FLA-backed model path.

## Expected Failure Modes

- No GPU assigned: free Colab did not give a GPU. Try later or use Colab Pro.
- `missing_triton`: the runtime or package set still cannot load Triton.
- `gdn_import_failed`: FLA installed, but its GatedDeltaNet layer did not import cleanly.

## Current Read

Pending Colab run.
