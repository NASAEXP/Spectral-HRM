# Experiment 1 - Fourier MLP Weights

## Goal

Test whether HRM-Text can replace the large dense MLP projection matrices with compact Fourier/DCT-style coefficient grids while keeping the rest of the text training stack unchanged.

This is the first clean check of the weight-system idea:

```text
same tokenizer
same data format
same HRM recurrence
same loss
only MLP Linear weights change
```

## What Changed

- Added `FourierLinear`, a compact DCT-style linear layer generated from trainable coefficient grids.
- Added `make_linear()` so dense `LinearInit` remains the default path.
- Added `fourier_linear` config support in `TransformerConfig`.
- Added `hrm_fourier.yaml` to enable Fourier only for MLP projections.
- Added focused tests for shape, gradient flow, parameter reduction, and config wiring.
- Added a slow PyTorch PrefixLM attention fallback for tiny local runs when FlashAttention-3 is unavailable.
- Made `LMHead` loss normalization work in single-process local runs without distributed init.
- Added `local_smoke.py`, a synthetic dense-vs-Fourier local training smoke script.

## Files Touched

- `models/layers.py`
- `models/transformer.py`
- `config/arch/net/hrm_fourier.yaml`
- `tests/test_fourier_linear.py`
- `tests/test_local_smoke.py`
- `tests/test_local_text_probe.py`
- `experiments/Experiment 1 - Fourier MLP Weights/local_smoke.py`
- `experiments/Experiment 1 - Fourier MLP Weights/local_text_probe.py`

## Current Config

```yaml
fourier_linear:
  enabled: true
  target: mlp
  in_modes: 256
  out_modes: 256
```

Attention, embeddings, and the LM head stay dense in this first experiment.

## Expected Parameter Effect

With `256 x 256` Fourier coefficient grids on MLP projections:

```text
B:  ~66x fewer trainable MLP parameters
L:  ~105x fewer trainable MLP parameters
XL: ~144x fewer trainable MLP parameters
```

This only describes MLP weights. It does not include embeddings, attention, LM head, optimizer state, activations, or runtime speed.

## Verification So Far

Commands run:

```bash
python -m pytest -q
python -m compileall models tests
python "experiments/Experiment 1 - Fourier MLP Weights/local_smoke.py" --steps 4 --device cpu
python "experiments/Experiment 1 - Fourier MLP Weights/local_smoke.py" --steps 8 --device cuda
python "experiments/Experiment 1 - Fourier MLP Weights/local_text_probe.py" --steps 24 --device cuda
python "experiments/Experiment 1 - Fourier MLP Weights/local_text_probe.py" --steps 120 --device cuda
python "experiments/Experiment 1 - Fourier MLP Weights/local_text_probe.py" --steps 120 --device cuda --modes 48
python "experiments/Experiment 1 - Fourier MLP Weights/local_mode_sweep.py" --steps 120 --device cuda --modes 16,24,32,48,64
```

Result:

```text
4 tests passed
compileall succeeded
tiny HRM construction sanity check found fourier_modules=4
CPU smoke:
  dense:   loss 6.1582 -> 2.6608, params=172,032
  fourier: loss 6.0400 -> 4.3380, params=74,752
CUDA smoke on RTX 3050 Ti:
  dense:   loss 6.1582 -> 0.5154, params=172,032, peak_vram_mb=19.5
  fourier: loss 6.0400 -> 2.8250, params=74,752, peak_vram_mb=19.3
CUDA local byte-text probe on repo text:
  24 steps, modes=24:
    dense:   eval 5.9695 -> 3.2957, params=289,536, peak_vram_mb=23.7
    fourier: eval 6.0657 -> 3.6602, params=144,384, peak_vram_mb=23.6
  120 steps, modes=24:
    dense:   eval 5.9695 -> 3.6999, params=289,536, peak_vram_mb=23.7
    fourier: eval 6.0657 -> 3.7607, params=144,384, peak_vram_mb=23.6
  120 steps, modes=48:
    dense:   eval 5.9695 -> 3.6999, params=289,536, peak_vram_mb=23.7
    fourier: eval 5.9447 -> 3.7275, params=151,296, peak_vram_mb=23.9
CUDA local mode sweep on repo byte-text, 120 steps:
  dense:
    eval 5.9695 -> 3.6999, params=289,536, peak_vram_mb=23.7
  fourier-16:
    eval 5.9892 -> 3.9543, params=143,104, peak_vram_mb=23.5
  fourier-24:
    eval 6.0657 -> 3.7607, params=144,384, peak_vram_mb=23.6
  fourier-32:
    eval 6.1188 -> 3.7791, params=146,176, peak_vram_mb=24.6
  fourier-48:
    eval 5.9447 -> 3.7275, params=151,296, peak_vram_mb=25.8
  fourier-64:
    eval 6.0103 -> 3.6528, params=158,464, peak_vram_mb=27.2
```

These smoke results use a tiny synthetic PrefixLM batch. They only prove the local code path runs and gradients can reduce loss. They do not measure real text quality.

The local byte-text probe uses repository text with a byte tokenizer, not Sapient's tokenizer or dataset. It is a better local signal than the synthetic batch, but still not a real HRM-Text data run.

In this tiny probe, `fourier-64` is the first configuration that slightly beats dense eval while using about 55% of the dense parameter count. Treat this as a promising local signal, not a quality claim.

## First Training Run

Planned first smoke:

```text
Dense HRM-small vs Fourier HRM-small
1M tokens first
then 10M tokens if loss moves
```

Track:

- GPU type
- wall time
- peak VRAM
- trainable parameters
- tokens/sec
- loss curve
- actual cost

## Open Questions

- Does the Fourier MLP path learn text at all?
- Are `256 x 256` modes too small, too large, or a good first default?
- Is the current factorized DCT computation fast enough for Colab testing?
- Does loss degrade gracefully as modes shrink?
