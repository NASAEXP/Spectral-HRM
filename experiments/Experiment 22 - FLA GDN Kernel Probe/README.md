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

```powershell
rtk python -m pip install -r requirements-fla.txt
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py"
```

Strict mode:

```powershell
rtk python "experiments\Experiment 22 - FLA GDN Kernel Probe\fla_gdn_probe.py" --require-ready
```

## Current Read

Local Windows result:

```text
status=missing_triton
fla_available=True
triton_available=False
gdn_import_ok=False
gdn_import_error=ModuleNotFoundError: No module named 'triton'
```

Read:

- FLA itself installs here.
- The optimized Gated DeltaNet layer does not import locally because Triton is unavailable for this Python/Windows environment.
- Do not wire `fla.layers.GatedDeltaNet` into HRM on this machine yet. The proper speed pass needs Linux/Colab where Triton can install and the FLA kernels can actually load.
