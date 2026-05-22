# Experiment 31 — Symmetry-compatible vocab optimizers

Compare **AdamW** vs **[equivariant_optimizers](https://github.com/timlautk/equivariant_optimizers)** on the **2D vocab/LM-head matrices** only (body stays AdamW).

## Stack

- **L:** PoM attention  
- **H:** FLA-GDN (`token_mixer=fla_gdn`, Fourier off)  
- **Heads:** `dense_tied`, `projected_dense_tied` (Exp 30 winners)

## Optimizers

| Label | Vocab optimizer | Notes |
|-------|-----------------|-------|
| `adamw` | AdamW | Baseline |
| `rownorm` | `RowNormM` | Cheap row-geometry; paper default for embed + LM head |
| `rightpolar` | `RightPolarGradM` | Right-spectral; `num_steps=5` Newton–Schulz on **H×H** Gram |
| `rightpolar-e10` | Polar every 10 steps, AdamW between | Throttle heavy polar |
| `hybrid` | `HybridPolarGradM` | Row + right spectral |

Repo is vendored at `.tools/equivariant_optimizers` (see `scripts/equivariant_vocab_optimizer.py`).

## Compute note

For weight `W` with shape `(V, H)` (e.g. 65536×256), **RightPolarGrad** forms `C = G^T G` with shape `(H, H)` — not an SVD on `V×H`. Cost per step is dominated by the `V×H` matmul for `G^T G` plus a few `H×H` polar iterations. **RowNormM** is cheaper (row scaling only).

## Results (40 steps, 3 seeds, PoM + FLA-GDN, May 2026)

| Label | mean eval ↓ |
|-------|------------:|
| projected-dense-tied + **adamw** | **3.17** |
| projected-dense-tied + rightpolar-e10 | 3.28 |
| dense-tied + adamw | 4.04 |
| dense-tied + rightpolar-e10 | 4.15 |
| projected + rownorm | 5.39 |
| projected + rightpolar (every step) | 6.13 |
| dense + rownorm | 6.39 |
| dense + hybrid | 6.99 |

Log: `runs/exp31/symmetry_vocab_20260522_191041.log`

At default `lr=2e-3`, symmetry optimizers did **not** beat AdamW on this probe. Full-step RowNorm / RightPolar / Hybrid were clearly worse; polar every 10 steps was close but still behind.

### RowNorm `lr_vocab` grid (dense-tied only, body `lr=2e-3`)

| `lr_vocab` | mean eval ↓ |
|-----------|------------:|
| **0.02** | **3.43** |
| 0.03 | 3.47 |
| 0.04 | 3.63 |
| 0.05 | 3.83 |
| 0.01 | 4.10 |

Best RowNorm (**3.43**) beats dense-tied + AdamW at matched `lr` (**4.04**), still above projected-dense-tied + AdamW (**3.17**).

```powershell
rtk python "experiments/Experiment 31 - Symmetry Vocab Optimizer/symmetry_vocab_optimizer_sweep.py" `
  --device cuda --heads spectral-dense-tied --vocab-optimizers rownorm `
  --lr 0.002 --lr-vocab-grid 0.01,0.02,0.03,0.04,0.05
```

## Run

```powershell
cd Spectral-HRM
rtk python "experiments/Experiment 31 - Symmetry Vocab Optimizer/symmetry_vocab_optimizer_sweep.py" --device cuda
```

Quick smoke (one seed, dense-tied only):

```powershell
rtk python "experiments/Experiment 31 - Symmetry Vocab Optimizer/symmetry_vocab_optimizer_sweep.py" --device cuda --seeds 1 --heads spectral-dense-tied --vocab-optimizers adamw,rownorm --steps 5
```
