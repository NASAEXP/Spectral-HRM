# Experiment 29 - Vocab Head Pareto Sweep

## Goal

Run the full vocab-head roadmap on a **CPU-friendly body** so local TDD and sweeps stay autonomous:

- **L-level:** `pom` + `fourier_linear` on L
- **H-level:** `attention` (no FLA-GDN)
- **Ordering:** `token_frequency` (from Experiment 14)

Compare **eval loss vs param count** across new head designs and micro-hybrid ranks.

## New head types (in `models/lm_head.py`)

| Config type | Idea |
| --- | --- |
| `hybrid_fourier_lowrank_asymmetric` | Fourier embed, Fourier + low-rank logits |
| `factorized_fourier` | 3-stage matmul, no dense `W` materialization |
| `multiscale_fourier` | Sum of several Fourier coefficient grids |
| `cluster_hybrid_fourier_lowrank` | Cluster-shared low-rank residual |
| `tiered_hot_fourier` | Fourier + dense rows for hot tokens |

## Variants

| Variant | Vocab head |
| --- | --- |
| `pom-dense-tied` | dense tied anchor |
| `pom-tied-fourier` | pure Fourier floor |
| `pom-factorized-fourier` | factorized Fourier |
| `pom-multiscale-fourier` | multiscale sum |
| `pom-tiered-hot-fourier` | hot-token tier |
| `pom-hybrid-r8` / `r16` / `r32` | tied hybrid bridge |
| `pom-asymmetric-hybrid-r16` | logits-only residual |
| `pom-cluster-hybrid-r16` | cluster residual |

## Tests

```powershell
rtk python -m pytest tests/test_vocab_head_designs.py tests/test_vocab_pareto_sweep.py -q
```

## How to run

CPU smoke:

```powershell
rtk python "experiments\Experiment 29 - Vocab Head Pareto Sweep\vocab_pareto_sweep.py" --steps 1 --warmup-steps 1 --seeds 1 --variants pom-tied-fourier,pom-asymmetric-hybrid-r16 --device cpu --hidden-size 64 --numseqs 2 --prefix-len 16 --causal-len 16 --eval-batches 1 --vocab-modes 16 --hidden-modes 8 --fourier-mode 8 --hot-token-count 8 --multiscale-specs "16,8;24,12"
```

Main sweep (local or Colab):

```powershell
rtk python "experiments\Experiment 29 - Vocab Head Pareto Sweep\vocab_pareto_sweep.py" --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda
```

Mode Pareto (fix rank cap on hidden axis):

```powershell
rtk python "experiments\Experiment 29 - Vocab Head Pareto Sweep\vocab_pareto_sweep.py" --steps 40 --seeds 1,2,3 --variants pom-tied-fourier --hidden-modes 64,128,192,256
```

Run once per `--hidden-modes` value and plot `mean_final_eval` vs `mean_num_params`.

## Recommended: SSH GPU (no notebook)

**Guide:** `docs/SSH_GPU_WORKFLOW.md`  
**Setup:** `scripts/remote_gpu_setup.sh`  
**Runs:** `scripts/run_exp29_followup.sh [attention|fla|all]`  

Rent a Linux GPU (Vast/RunPod/Lightning), connect with **Cursor Remote SSH**, run shell scripts only.

## Colab / Kaggle (optional legacy)

**Notebook:** `colab/experiment_29_followup.ipynb`  
**Guide:** `COLAB.md` / `kaggle/KAGGLE.md`

1. Push repo → Colab GPU → run FLA gate (Exp 22).
2. `colab_followup_sweep.py` — `hidden_modes` 64/128/192/256 on multiscale + tiered-hot + tied baseline.
3. `fla_gdn_port_sweep.py` — same heads on **PoM L + FLA-GDN H** (Exp 26/28 comparable).

Paste Colab `summary:` output back to lock presets.

## Read

Pick the **Pareto knee**: best eval for a param budget, not the global minimum at dense-tied scale.

## Results (local CUDA, 2026-05-22)

Config: `steps=40`, `warmup=1`, `seeds=1,2,3`, `hidden=256`, `128×128`, `vocab_modes=512`, `hidden_modes=64`, `multiscale_specs=512,64;256,128`, body **PoM L + attention H** (no FLA-GDN).

| Variant | Mean eval ↓ | Params |
| --- | ---: | ---: |
| `pom-dense-tied` | **5.3576** | 17,518,592 |
| `pom-tiered-hot-fourier` | **6.7437** | 1,888,256 |
| `pom-multiscale-fourier` | 6.8899 | 872,448 |
| `pom-hybrid-r32` | 6.8809 | 2,945,024 |
| `pom-hybrid-r16` | 6.9876 | 1,892,352 |
| `pom-asymmetric-hybrid-r16` | 6.9902 | 1,892,352 |
| `pom-cluster-hybrid-r16` | 6.9962 | 847,872 |
| `pom-hybrid-r8` | 7.0009 | 1,366,016 |
| `pom-tied-fourier` | 7.0311 | 839,680 |
| `pom-factorized-fourier` | 7.0311 | 839,680 |

Read:

- **Quality anchor:** `pom-dense-tied` still wins by ~1.4 eval points.
- **Best sub-1M params:** `pom-multiscale-fourier` (~6.89) beats pure `pom-tied-fourier` (~7.03) at similar size.
- **Best ~2M params:** `pom-tiered-hot-fourier` (~6.74); hot-token tier is the clearest new-head win.
- **Hybrids / asymmetric / cluster:** no meaningful gain over multiscale or tiered-hot at this step budget; `factorized` matches tied (same math, different forward).
- Full log: `experiments/Experiment 29 - Vocab Head Pareto Sweep/run_output.txt`

## Results (local CUDA + FLA-GDN, 2026-05-22)

Config: same as above, body **PoM L + FLA-GDN H** (`fla_gdn_port_sweep.py`), `triton-windows` on Windows. ~5.5 min. Log: `runs/exp29/fla_port_20260522_181505.log`

| Variant | Mean eval ↓ | Params |
| --- | ---: | ---: |
| `spectral-dense-tied` | **4.0373** | 17,523,784 |
| `spectral-tiered-hot-fourier` | **5.9450** | 1,893,448 |
| `spectral-multiscale-fourier` | 6.7422 | 877,640 |
| `spectral-tied-fourier` | 6.9802 | 844,872 |

Read vs attention-only (PoM L + attention H):

- **Dense-tied:** FLA body **~1.3 eval better** (4.04 vs 5.36).
- **Tiered-hot:** FLA **~0.8 eval better** (5.95 vs 6.74).
- **Multiscale / tied-fourier:** ~flat (~6.74–6.98 vs ~6.89–7.03).
- Peak VRAM (reported): ~1.8–2.4 GB per variant — fits local GPU with clock cap.

## Results (FLA-GDN hidden_modes sweep, 2026-05-22)

Variants: `spectral-multiscale-fourier`, `spectral-tiered-hot-fourier`. `hidden_modes` ∈ {64, 128, 192, 256}, matched `multiscale_specs`. ~5.2 min total. Log: `runs/exp29/fla_hidden_modes_20260522_182201.log`

| hidden_modes | multiscale eval ↓ | tiered-hot eval ↓ |
| ---: | ---: | ---: |
| 64 | 6.7422 | 5.9450 |
| 128 | 6.8507 | **5.8552** |
| 192 | 6.8782 | 5.9307 |
| 256 | **6.5539** | 6.1405 |

Read:

- **`spectral-tiered-hot-fourier`:** best at **128** (5.86); 256 hurts (~6.14).
- **`spectral-multiscale-fourier`:** best at **256** (6.55); 64–192 do not beat that knee.
- Raising `hidden_modes` is not monotonic — pick per head, not globally.

Preset candidates: `tiered-hot` @ `hidden_modes=128`; `multiscale` @ `hidden_modes=256` (FLA body).
