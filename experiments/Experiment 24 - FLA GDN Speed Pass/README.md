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

Free Colab T4 smoke:

```python
!python "experiments/Experiment 24 - FLA GDN Speed Pass/fla_gdn_speed_pass.py" --steps 5 --warmup-steps 1 --seeds 1 --device cuda
```

| Variant | H-level | Eval Loss | Params | Peak VRAM | Warmup | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-sla` | SLA | 7.5673 | 135,684 | 330.6 MB | 0.22 s | 20.49 | 9,369.3 | fastest control |
| `pom-gdn` | local GDN | 8.0882 | 140,296 | 334.0 MB | 0.29 s | 201.11 | 954.7 | local recurrence bottleneck |
| `pom-fla-gdn` | FLA GDN | 7.4979 | 301,608 | 439.6 MB | 268.74 s | 21.04 | 9,124.3 | compile-heavy, then fast |

Read:

- The earlier `53s/step` result was compile cost, not steady-state speed.
- FLA GDN has a huge first-run warmup on free Colab/T4: `268.74s`.
- After warmup, FLA GDN is basically SLA-speed: `21.04 ms/step` vs `20.49 ms/step`.
- FLA GDN is roughly `9.6x` faster than local GDN on measured train steps.
- FLA GDN uses more parameters and VRAM in this first wrapper: `301,608` params and `439.6 MB`.

Free Colab T4 40-step, 3-seed result:

```python
!python "experiments/Experiment 24 - FLA GDN Speed Pass/fla_gdn_speed_pass.py" --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda
```

| Variant | H-level | Mean Eval Loss | Params | Peak VRAM | Warmup | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-sla` | SLA | 6.8143 | 135,684 | 330.6 MB | 0.08 s | 17.71 | 10,842.5 | fastest and smallest |
| `pom-gdn` | local GDN | 6.7971 | 140,296 | 334.0 MB | 0.25 s | 246.84 | 783.8 | slight quality gain, too slow |
| `pom-fla-gdn` | FLA GDN | 6.7780 | 301,608 | 366.1 MB | 91.01 s | 20.48 | 9,475.5 | best loss, near SLA speed |

Read:

- `pom-fla-gdn` is the best mean eval loss in this run: `6.7780`.
- It is much faster than local GDN: `20.48 ms/step` vs `246.84 ms/step`, about `12x`.
- It is close to SLA speed: `20.48 ms/step` vs `17.71 ms/step`.
- Its first seed paid the big Triton compile cost, so mean warmup is still high at `91.01s`.
- It uses more params than the local mixers because FLA GDN owns dense internal projections.
- Current read: `pom-fla-gdn` is now worth keeping as the H-level speed/quality candidate, but we need a larger-context run before locking it.

Free Colab T4 larger-context result:

```python
!python "experiments/Experiment 24 - FLA GDN Speed Pass/fla_gdn_speed_pass.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda \
  --hidden-size 256 \
  --numseqs 8 \
  --prefix-len 128 \
  --causal-len 128
```

| Variant | H-level | Mean Eval Loss | Params | Peak VRAM | Warmup | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pom-sla` | SLA | 7.0822 | 136,196 | 1,943.4 MB | 0.14 s | 109.08 | 18,885.3 | fast control, worse loss |
| `pom-gdn` | local GDN | 6.9962 | 141,320 | 2,077.2 MB | 2.58 s | 2,704.62 | 758.9 | better loss, unusably slow |
| `pom-fla-gdn` | FLA GDN | 6.8271 | 844,872 | 1,829.2 MB | 57.34 s | 100.60 | 20,358.3 | best loss and fastest measured tok/s |

Read:

- Larger context strengthens the FLA GDN result.
- `pom-fla-gdn` beats SLA by `0.2551` mean eval loss: `6.8271` vs `7.0822`.
- `pom-fla-gdn` is slightly faster than SLA on measured token throughput: `20,358 tok/s` vs `18,885 tok/s`.
- `pom-fla-gdn` uses less peak VRAM than SLA in this run: `1,829.2 MB` vs `1,943.4 MB`.
- `pom-fla-gdn` has far more parameters because the FLA H-level owns dense projection weights: `844,872` vs `136,196`.
- Local GDN remains a correctness/reference path, not a practical speed path.
- Current read: for bigger context, `pom-fla-gdn` is the best H-level candidate so far.
