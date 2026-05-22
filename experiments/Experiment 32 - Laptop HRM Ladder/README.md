# Experiment 32 — Laptop HRM ladder

## Goal

Long training on **real HRM-Text data** (5M-token laptop slice), scaling `hidden_size`, to see if the **projected-dense-tied + PoM + FLA-GDN** recipe still beats the **original-ish untied dense-attention** path after more steps and larger models — the closest laptop comparison to a cloud pretrain before spending real budget.

## What changed

- Added `laptop_hrm_ladder.py` (wraps Exp 25 `train_once` / `make_config`).
- Uses `data_io/data_laptop_hrm_slice` (real cleaned Sapient JSONL → BPE).
- Compares `fourier-pom-fla-gdn-projected-dense-tied` vs `dense-attention` only.

## Files touched

- `experiments/Experiment 32 - Laptop HRM Ladder/laptop_hrm_ladder.py`
- `experiments/Experiment 32 - Laptop HRM Ladder/README.md`
- `data_io/build_laptop_hrm_slice.py`, `data_io/LAPTOP_SLICE.md` (data for all `hrm_slice` runs)
- `Spectral-HRM/scripts/load_probe_tokens.py`

## How to run

Build slice once (if missing):

```powershell
cd C:\Users\Dos\Documents\GRAM\data_io
python build_laptop_hrm_slice.py
```

Main ladder (logged run below):

```powershell
cd C:\Users\Dos\Documents\GRAM\Spectral-HRM
$env:PYTHONUNBUFFERED = "1"
python -u "experiments/Experiment 32 - Laptop HRM Ladder/laptop_hrm_ladder.py" --device cuda --steps 500 --seeds 1,2,3 --hidden-sizes 256,384,512 *>&1 | Tee-Object runs/exp32/ladder_500.log
```

Defaults: `steps=500`, `seeds=1,2,3`, `hidden_sizes=256,384,512`, `eval_batches=8`, `data=hrm_slice`.

Smoke:

```powershell
python -u "experiments/Experiment 32 - Laptop HRM Ladder/laptop_hrm_ladder.py" --device cuda --steps 50 --seeds 1 --hidden-sizes 256 --variants fourier-pom-fla-gdn-projected-dense-tied
```

## Run shape (completed 2026-05-22, local CUDA)

- **Data:** `hrm_slice` — 4,999,938 tokens (train 3,999,950 / eval 999,988), gsm8k + math + no_robots + 25k webinstruct
- **Steps:** 500, warmup 1, seeds 1–3
- **Context:** 128×128, `numseqs=8`, AdamW `lr=2e-3`
- **Wall time:** ~44 min (18 runs)
- **Log:** `runs/exp32/ladder_500_20260522_195845.log`

## Results — mean final eval ↓ (lower is better)

| Variant | h=256 | h=384 | h=512 |
| --- | ---: | ---: | ---: |
| **`fourier-pom-fla-gdn-projected-dense-tied`** | **4.6126** | **4.5747** | **4.5385** |
| `dense-attention` (original-ish untied) | 4.9229 | 4.9311 | 4.9382 |

Stdev across seeds: ~0.015–0.027 (tight).

### Params and VRAM (means)

| Variant | h=256 | h=384 | h=512 |
| --- | ---: | ---: | ---: |
| Projected + PoM/FLA params | 17.65M | 26.73M | 36.40M |
| Untied dense-attention params | 35.00M | 52.99M | 72.09M |
| Projected peak VRAM (MB) | 1892 | 2173 | 2453 |
| Untied peak VRAM (MB) | 2052 | 2284 | 2545 |

### Throughput (tokens/s, train phase, indicative)

| h | projected | untied |
| ---: | ---: | ---: |
| 256 | ~11.7k | ~11.4k |
| 384 | ~8.5k | ~8.0k |
| 512 | ~7.5k | ~6.7k |

## Read

- **Projected tie wins at every hidden size** (~0.31–0.40 eval better than untied) after **500 steps** on real data — stronger than the 40-step slice (Exp 25: ~5.68 vs ~5.84).
- **Scaling h 256→512** barely moves untied eval (~4.92→4.94); **projected improves** (~4.61→4.54). Bigger probe body helps the spectral stack more than the original-ish path.
- **Param ratio holds:** ~half the params of untied at each h; at h=512 projected **36M vs 72M**.
- Still **not** 0.6B / 40B-token pretrain — see `data_io/LAPTOP_SLICE.md` and cost notes in conversation; next step is **$100 cloud pilot** (B-scale, ~3–5B tokens) with `pretrain.py`.

## Comparison to earlier probes

| Run | Steps | Data | Projected eval | Untied eval |
| --- | ---: | --- | ---: | ---: |
| Exp 25 `hrm_slice` | 40 | 5M slice | 5.68 | 5.84 |
| **Exp 32 ladder h=256** | **500** | 5M slice | **4.61** | **4.92** |
| Exp 30 projected (README probe) | 40 | README BPE | 3.17 | — |
| Exp 25 full stack h=256 | 40 | README BPE | 3.17 | 3.55 |

Real slice + longer train **increases** loss (harder data) but **widens** the projected-vs-untied gap on the slice.

## Open questions

- Does projected tie still win on **V1Dataset** batches (true inst/resp packing) vs flat sliding windows?
- 1× GPU **tokens/$** bench on size B before renting cloud time?
- Official **`evaluation.main`** on Sapient L checkpoint vs a laptop-trained B checkpoint (reference ceiling)?

## Downstream benchmarks (GSM8k / MATH) on saved probe weights

Training does not write FSDP checkpoints; use `--save-ckpt-dir` to export probe weights compatible with `scripts/probe_benchmark.py`.

Train + save (example: one seed, h=512, both ladder variants):

```powershell
cd C:\Users\Dos\Documents\GRAM\Spectral-HRM
python -u "experiments/Experiment 32 - Laptop HRM Ladder/laptop_hrm_ladder.py" --device cuda --steps 500 --seeds 1 --hidden-sizes 512 `
  --save-ckpt-dir runs/exp32/ckpts
```

GSM8k smoke (50 problems, CoT condition — same as `hrm_benchmarking.yaml` math tasks):

```powershell
python scripts/probe_benchmark.py --ckpt-dir runs/exp32/ckpts/fourier-pom-fla-gdn-projected-dense-tied_h512_s1 `
  --benchmarks GSM8k --limit 50 --batch-size 4 --max-context 2048 --max-tokens 512

python scripts/probe_benchmark.py --ckpt-dir runs/exp32/ckpts/dense-attention_h512_s1 `
  --benchmarks GSM8k --limit 50 --batch-size 4
```

Compare projected vs untied on the same slice-trained weights. Absolute **acc** will be low (tiny model, 500 steps) — use for **relative** ranking only.

Full MATH (slow): `--benchmarks MATH --limit 0` (0 = entire test set).

Checkpoint layout: `model.pt`, `model_config.json`, `probe_meta.json`, `tokenizer_info.json`, `token_permutation.pt`.

## Optional — official reference eval

```powershell
python -m evaluation.main ckpt_path="sapientinc/HRM-Text-1B" run_only=[GSM8k,MATH] generation_config.batch_size=4
```

Requires sufficient GPU RAM; not comparable param count to this experiment.
