# Experiment 25 - Full Stack Comparison

## Goal

Compare the original-ish dense HRM-Text path against the current Spectral-HRM stack in one fair script.

This is the check before we make stronger claims about cost. Experiment 24 showed that FLA GDN is fast after warmup, but it did not include the dense attention control or the tied vocab choices in the same run.

## What changed

Added `full_stack_comparison.py` with these variants:

| Variant | L-level | H-level | Vocab/head | Why it is here |
| --- | --- | --- | --- | --- |
| `dense-attention` | dense attention | dense attention | untied dense vocab | original-ish control |
| `fourier-pom-sla-tied-fourier` | PoM | SLA | tied Fourier vocab | current small/fast spectral baseline |
| `fourier-pom-fla-gdn-tied-fourier` | PoM | FLA GDN | tied Fourier vocab | best compressed-vocab GDN candidate |
| `fourier-pom-fla-gdn-dense-tied` | PoM | FLA GDN | dense tied vocab | checks if the Fourier vocab is helping or hurting |

For FLA GDN, the H-level disables `FourierLinear` projections because the optimized FLA wrapper owns dense internal projections.

## Files touched

- `experiments/Experiment 25 - Full Stack Comparison/full_stack_comparison.py`
- `tests/test_full_stack_comparison.py`
- `colab/free_fla_gdn_probe.ipynb`
- `tests/test_free_colab_probe_notebook.py`
- `experiments/README.md`

## How to run

Fast Colab smoke:

```python
%cd /content/Spectral-HRM
!python "experiments/Experiment 25 - Full Stack Comparison/full_stack_comparison.py" \
  --steps 5 \
  --warmup-steps 1 \
  --seeds 1 \
  --device cuda
```

Main comparison:

```python
%cd /content/Spectral-HRM
!python "experiments/Experiment 25 - Full Stack Comparison/full_stack_comparison.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Local non-FLA smoke:

```powershell
rtk python "experiments\Experiment 25 - Full Stack Comparison\full_stack_comparison.py" --steps 1 --warmup-steps 1 --seeds 1 --variants dense-attention,fourier-pom-sla-tied-fourier --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16
```

## Verification

Focused tests:

```powershell
rtk python -m pytest tests/test_full_stack_comparison.py tests/test_free_colab_probe_notebook.py -q
```

Local non-FLA smoke:

```powershell
rtk python "experiments\Experiment 25 - Full Stack Comparison\full_stack_comparison.py" --steps 1 --warmup-steps 1 --seeds 1 --variants dense-attention,fourier-pom-sla-tied-fourier --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16 --eval-batches 1
```

Broader checks:

```powershell
rtk python -m compileall experiments tests colab
rtk git diff --check
```

## Current read

No Experiment 25 Colab result yet.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-attention` | 11.4412 | 8,527,872 | 125.66 | 509.3 | dense original-ish control |
| `fourier-pom-sla-tied-fourier` | 8.0017 | 135,428 | 239.85 | 266.8 | compressed spectral path |

This local smoke is only a wiring check. The real comparison still needs Colab with CUDA, especially for the FLA GDN variants.

The expected decision point is simple:

- If `dense-attention` is much better per token, we have more architecture work before cost claims.
- If the spectral variants are close while using fewer trained vocab/body parameters, the cost path is still alive.
- If `fourier-pom-fla-gdn-tied-fourier` beats or matches `fourier-pom-fla-gdn-dense-tied`, the Fourier vocab remains worth keeping.

## Open questions

- Does the dense attention control win on loss once the context and hidden size are large enough?
- Is FLA GDN still worth its extra parameters when compared against the full spectral baseline?
- Does tied Fourier vocab remain competitive against dense tied vocab at larger scale?
