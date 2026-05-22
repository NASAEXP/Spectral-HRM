# Experiment 30 - Non-Fourier Vocab Heads

## Goal

Sweep **LM head / embedding** designs **without Fourier vocab** on the locked trunk:

- **L:** PoM
- **H:** FLA-GDN

## New head types (`models/lm_head.py`)

| Type | Idea |
| --- | --- |
| `projected_dense_tied` | \(W_{out} = W_{in} T\), small `T` (256×256) |
| `tied_lowrank` | \(W \approx A B\), tied embed/logits |
| `tiered_hot_lowrank` | low-rank cold bulk + dense hot rows |
| `tied_kronecker` | \(W = A \otimes B\) with 256×256 vocab factors |
| `untied_lowrank` | separate low-rank embed and logits |
| `untied_dense` | separate full dense embed and logits |

Anchor: `dense_tied` as `spectral-dense-tied`.

## Tests

```powershell
rtk python -m pytest tests/test_non_fourier_vocab_heads.py tests/test_non_fourier_vocab_sweep.py -q
```

## Run

```powershell
rtk python "experiments\Experiment 30 - Non-Fourier Vocab Heads\non_fourier_vocab_sweep.py" --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda --tokenizer-path "C:\Users\Dos\Documents\GRAM\data_io\trained_tokenizers\bpe\tokenizer.json"
```

Smoke:

```powershell
rtk python "experiments\Experiment 30 - Non-Fourier Vocab Heads\non_fourier_vocab_sweep.py" --steps 1 --seeds 1 --variants spectral-dense-tied,spectral-projected-dense-tied --device cuda
```

Logs: `runs/exp30/`

## Results (local CUDA + FLA-GDN, 2026-05-22)

Config: `steps=40`, `warmup=1`, `seeds=1,2,3`, `hidden=256`, `128×128`, `lowrank_rank=128`, `hot_token_count=4096`, Kronecker factors `256×256` / `16×16`. Log: `runs/exp30/non_fourier_*.log`

| Variant | Mean eval ↓ | Params |
| --- | ---: | ---: |
| **spectral-projected-dense-tied** | **3.1728** | 17,654,856 |
| spectral-untied-dense | 4.0277 | 34,366,536 |
| spectral-dense-tied | 4.0373 | 17,523,784 |
| spectral-tiered-hot-lowrank | 4.4834 | 10,282,056 |
| spectral-tied-lowrank | 4.4563 | 9,233,480 |
| spectral-untied-lowrank | 4.4563 | 17,654,856 |
| spectral-tied-kronecker | 7.7941 | 820,296 |

Read:

- **Projected tie (`W_out = W_in @ T`) beats strict dense-tied** by ~0.86 eval with only +131k params (`T` is 256×256). Best head in this repo so far on the probe.
- **Tiered-hot + low-rank cold** (~4.48) beats **tiered-hot-fourier** (~5.9, Exp 29) at ~10M params — dropping Fourier on the cold bulk helped.
- **Tied low-rank @ r=128** (~4.46, ~9.2M) beats Fourier floor (~7) and hybrid ~6; **untied low-rank** same eval but ~2× head params — skip untied low-rank.
- **Kronecker** still ~7.8 — same failure mode as tied-fourier (too rigid), even with tiny param count.
- **Untied dense** ≈ dense-tied (~4.03) at **2×** vocab params — not worth vs projected tie.

**Follow-up:** [Experiment 32](../Experiment%2032%20-%20Laptop%20HRM%20Ladder/README.md) — 500 steps on **hrm_slice**, h=256–512: projected **4.54–4.61** vs untied **4.92–4.94**. [Experiment 31](../Experiment%2031%20-%20Symmetry%20Vocab%20Optimizer/README.md) — symmetry optimizers on dense-tied only; projected + AdamW still best at probe scale.
