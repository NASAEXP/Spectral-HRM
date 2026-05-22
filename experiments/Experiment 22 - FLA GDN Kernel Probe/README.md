# Experiment 22 - FLA GDN Kernel Probe

## Goal

Check whether the optimized FLA Gated DeltaNet path can run in the current local environment.

This is separate from Experiment 21 on purpose:

```text
Experiment 21 = local exact GDN behavior
Experiment 22 = can we use the optimized FLA kernel path here?
```

## What Changed

- Installed `flash-linear-attention==0.5.0` locally.
- Added `requirements-fla.txt` so the optional FLA dependency is explicit without forcing it into the base HRM-Text install.
- Added a small import probe for `fla`, `triton`, and `fla.layers.GatedDeltaNet`.
- Kept this out of the HRM model path until the optimized kernel imports cleanly.

## Sources

- [FLA repo](https://github.com/fla-org/flash-linear-attention)
- [FLA PyPI package](https://pypi.org/project/flash-linear-attention/)
- [NVLabs GatedDeltaNet repo](https://github.com/NVlabs/GatedDeltaNet)

## How To Run

### Windows (native, no WSL)

Use [triton-windows](https://github.com/triton-lang/triton-windows) — not `colab/install_fla_colab.py` (Linux wheels).

```powershell
rtk python -m pip uninstall -y triton pytorch-triton pytorch-triton-rocm
rtk python -m pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
rtk python -m pip install -U "triton-windows<3.8"
rtk python -m pip install -r requirements-fla.txt
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py" --require-ready
```

Match `triton-windows<3.x` to your PyTorch minor version (see triton-windows README).

### Linux / Colab

```powershell
rtk python colab/install_fla_colab.py
```

Or:

```powershell
rtk python -m pip install -r requirements-fla.txt
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py"
```

Strict mode:

```powershell
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py" --require-ready
```

## Current Read

Local Windows result (2026-05-22, `triton-windows` 3.7.0, `torch` 2.12+cu126):

```text
status=ready
fla_available=True
triton_available=True
gdn_import_ok=True
```

Read:

- FLA GatedDeltaNet imports on Windows when `triton-windows` is installed.
- You can run FLA-GDN port sweeps locally (`fla_gdn_port_sweep.py`, `run_exp29_followup.sh fla`) without Colab/Linux for Triton alone.

Free Colab result:

```text
status=ready
fla_available=True
triton_available=True
gdn_import_ok=True
```

Read:

- Free Colab can import the FLA GatedDeltaNet layer.
- This unlocks Experiment 24 as the real optimized speed pass.
