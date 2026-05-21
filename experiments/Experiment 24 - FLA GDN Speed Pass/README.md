# Experiment 24 - FLA GDN Speed Pass

## Goal

Now that free Colab reports `status=ready` for FLA/Triton, test the optimized FLA Gated DeltaNet path against our current H-level controls:

```text
L-level = PoM
H-level = SLA vs local GDN vs FLA GDN
context = 48x48
```

## What Changed

- Added `FLAGatedDeltaNetAttention`, a thin adapter from HRM's packed `[tokens, hidden]` format to FLA's `[batch, seq, hidden]` format.
- Added `token_mixer="fla_gdn"`.
- Added this runner for Colab/Linux speed comparison.
- Disabled FourierLinear only inside the `pom-fla-gdn` H-level override, because FLA uses its own dense projections.

## Important Note

This wrapper currently requires equal-length packed sequences. That matches our current experiment runner, where every batch uses the same `prefix_len + causal_len`.

It also uses FLA's causal GDN behavior for the whole sequence. That means it is a speed-path candidate first; quality must be measured against `pom-sla` and local `pom-gdn`.

For `pom-fla-gdn`, the L-level and vocab stack still use the Fourier survival preset. The H-level FLA mixer does not, because the FLA layer owns its internal projection weights.

## How To Run On Colab

After the Experiment 22 cell prints `status=ready`, run:

```python
!python "experiments/Experiment 24 - FLA GDN Speed Pass/fla_gdn_speed_pass.py" --steps 5 --warmup-steps 1 --seeds 1 --device cuda
```

If that passes, run the 40-step comparison:

```python
!python "experiments/Experiment 24 - FLA GDN Speed Pass/fla_gdn_speed_pass.py" --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `pom-sla` | PoM | SLA | current fast H control |
| `pom-gdn` | PoM | local GDN | exact local Python-level GDN |
| `pom-fla-gdn` | PoM | FLA GDN | optimized FLA/Triton-backed GDN |

## Current Read

Pending Colab run.
