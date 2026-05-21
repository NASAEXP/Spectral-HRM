# Experiment 19 - Long Context H-Level Validation

## Goal

Check whether SLA's H-level advantage grows when sequence length gets longer.

Experiment 18 showed SLA barely ahead of attention at short context. This run keeps the same model stack and sweeps context size:

```text
L-level = PoM
H-level = attention vs SLA vs SPECTRE
contexts = 12x12, 24x24, 48x48
```

## What Changed

- Added a context-sweep runner on top of Experiment 18.
- Kept the H-level candidates identical:
  - `pom-attention`
  - `pom-sla`
  - `pom-spectre`
- Tracked loss, VRAM, total time, training ms/step, and training tokens/sec per context.

## How To Run

```powershell
rtk python "experiments\Experiment 19 - Long Context H-Level Validation\long_context_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

Fast smoke:

```powershell
rtk python "experiments\Experiment 19 - Long Context H-Level Validation\long_context_h_level.py" --steps 5 --seeds 1 --contexts 12x12 --device cuda
```

## Variants

| Variant | L-level | H-level | Meaning |
| --- | --- | --- | --- |
| `pom-attention` | PoM | attention | current H control |
| `pom-sla` | PoM | SLA | current H candidate |
| `pom-spectre` | PoM | SPECTRE | spectral H control |

## Current Read

Command:

```powershell
rtk python "experiments\Experiment 19 - Long Context H-Level Validation\long_context_h_level.py" --steps 40 --seeds 1,2,3 --device cuda
```

RTX 3050 Ti local result:

| Context | Variant | Mean Eval Loss | Peak VRAM | Train ms/step | Train tok/s | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `12x12` | `pom-attention` | 6.8123 | 243.6 MB | 18.15 | 2,644.3 | short control |
| `12x12` | `pom-sla` | 6.7913 | 243.6 MB | 19.16 | 2,505.7 | SLA wins short context |
| `12x12` | `pom-spectre` | 6.8983 | 244.2 MB | 75.40 | 637.4 | slower and worse |
| `24x24` | `pom-attention` | 7.1848 | 252.8 MB | 19.39 | 4,951.6 | attention wins mid context |
| `24x24` | `pom-sla` | 7.1936 | 254.4 MB | 20.78 | 4,620.9 | tiny loss gap |
| `24x24` | `pom-spectre` | 7.2078 | 256.0 MB | 131.87 | 729.2 | slower and worse |
| `48x48` | `pom-attention` | 6.2906 | 326.5 MB | 19.86 | 9,697.4 | long control |
| `48x48` | `pom-sla` | 6.2687 | 330.3 MB | 19.65 | 9,773.1 | SLA wins long context |
| `48x48` | `pom-spectre` | 6.2692 | 337.8 MB | 244.14 | 787.0 | quality close, speed bad |

Read:

- SLA does not dominate every context, but it wins at `12x12` and `48x48`.
- At `24x24`, attention wins by only `0.0088` eval loss, which is too small to treat as a strong rejection of SLA.
- At `48x48`, SLA slightly beats attention and is also a bit faster in this run.
- SPECTRE becomes quality-competitive at `48x48`, but its training loop is roughly an order of magnitude slower.
- Keep SLA as the H-level frontrunner, but do not hard-lock it yet. Next candidate family should be GLA/GSA against `pom-sla`.
