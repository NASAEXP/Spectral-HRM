# Experiment 25 - Full Stack Comparison

## Goal

Compare the original-ish dense HRM-Text path against the current Spectral-HRM stack in one fair script.

This is the check before we make stronger claims about cost. Experiment 24 showed that FLA GDN is fast after warmup, but it did not include the dense attention control or the tied vocab choices in the same run.

## What changed

Added `full_stack_comparison.py` with these variants:

| Variant | L-level | H-level | Vocab/head | Why it is here |
| --- | --- | --- | --- | --- |
| `dense-attention` | dense attention | dense attention | untied dense vocab | original-ish control |
| `dense-tied-attention` | dense attention | dense attention | dense tied vocab | separates attention quality from untied vocab/head capacity |
| `fourier-pom-sla-tied-fourier` | PoM | SLA | tied Fourier vocab | current small/fast spectral baseline |
| `fourier-pom-fla-gdn-tied-fourier` | PoM | FLA GDN | tied Fourier vocab | best compressed-vocab GDN candidate |
| `fourier-pom-fla-gdn-dense-tied` | PoM | FLA GDN | dense tied vocab | checks if the Fourier vocab is helping or hurting |
| `fourier-pom-fla-gdn-projected-dense-tied` | PoM | FLA GDN | projected dense tied (`W_out = W_in @ T`) | current Spectral-HRM quality baseline |

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

### Colab T4 main comparison

Command:

```python
!python "experiments/Experiment 25 - Full Stack Comparison/full_stack_comparison.py" \
  --steps 40 \
  --warmup-steps 1 \
  --seeds 1,2,3 \
  --device cuda
```

Run shape:

- `tokens=115,170`
- `train=92,136`
- `eval=23,034`
- `context=128x128`
- `hidden_size=256`
- `numseqs=8`
- `vocab_size=65,536`

Summary:

| Variant | Final eval | Params | Peak VRAM | ms/step | Tokens/s | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dense-attention` | 2.4711 | 34,996,224 | 2,052.5 MB | 92.46 | 22,172.0 | best loss, but has untied dense vocab/head |
| `dense-tied-attention` | 5.7842 | 18,219,008 | 1,860.5 MB | 87.78 | 23,345.2 | tying the dense vocab/head causes a large loss drop |
| `fourier-pom-sla-tied-fourier` | 7.0822 | 136,196 | 1,943.4 MB | 102.37 | 20,021.4 | very small, but loss is too high |
| `fourier-pom-fla-gdn-tied-fourier` | 6.8271 | 844,872 | 1,829.2 MB | 91.90 | 22,324.4 | fast and compact, but still too compressed |
| `fourier-pom-fla-gdn-dense-tied` | 3.4149 | 17,523,784 | 1,827.5 MB | 88.64 | 23,125.0 | best spectral tradeoff |

Plain read:

- The original-ish `dense-attention` win depends heavily on the untied dense vocab/head. When we tie the dense vocab/head, eval drops from `2.4711` to `5.7842`.
- Under the tied dense vocab/head condition, our spectral body wins: `fourier-pom-fla-gdn-dense-tied` reaches `3.4149` with slightly fewer params, lower VRAM, and similar or better speed.
- Fully Fourier-tied vocab is still too aggressive here. It cuts params hard, but the loss gap is large.
- FLA GDN remains worth keeping as the H-level candidate. The useful stack right now is `PoM L-level + FLA GDN H-level + dense tied vocab/head`.
- This still does not prove the `~$100` training target. It does show the cost path is not dead: the spectral body can beat dense tied attention, but the vocab/head compression needs a gentler bridge than pure Fourier.

Next gate:

- Add a vocab bridge sweep between `tied_fourier` and `dense_tied`: larger modes, hybrid dense residual, or trainable low-rank residual.
- Keep `dense-attention`, `dense-tied-attention`, and `fourier-pom-fla-gdn-dense-tied` as the three anchors.
- Do not optimize the local GDN path further right now; FLA GDN is the path that matters on Colab/Linux.

Tiny local CPU smoke:

| Variant | Final eval | Params | ms/step | Tokens/s | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense-attention` | 11.4412 | 8,527,872 | 125.66 | 509.3 | dense original-ish control |
| `dense-tied-attention` | 11.5891 | 4,333,568 | 110.39 | 579.8 | dense attention with shared vocab/head |
| `fourier-pom-sla-tied-fourier` | 8.0017 | 135,428 | 239.85 | 266.8 | compressed spectral path |

This local smoke is only a wiring check. The real comparison still needs Colab with CUDA, especially for the FLA GDN variants.

The expected decision point is simple now:

- `dense-attention` is the full-capacity quality anchor.
- `dense-tied-attention` shows how expensive vocab/head tying is for a normal dense body.
- `fourier-pom-fla-gdn-dense-tied` shows the spectral body can recover much of that tied-vocab loss.
- The next win needs to come from the vocab/head bridge, not another H-level mixer search.

## Projected baseline vs original HRM (local CUDA, 2026-05-22)

Same script/harness as above (`steps=40`, `seeds=1,2,3`, BPE 65k, `128×128`).

```powershell
rtk python "experiments/Experiment 25 - Full Stack Comparison/full_stack_comparison.py" `
  --device cuda --steps 40 --seeds 1,2,3 `
  --variants dense-attention,dense-tied-attention,fourier-pom-fla-gdn-dense-tied,fourier-pom-fla-gdn-projected-dense-tied
```

| Variant | Mean eval ↓ | Params | Read |
| --- | ---: | ---: | --- |
| **`fourier-pom-fla-gdn-projected-dense-tied`** | **3.17** | 17,654,856 | **best** — PoM + FLA-GDN + projected tie |
| `dense-attention` (original-ish) | 3.55 | 34,996,224 | untied embed + LM head (~2× vocab capacity) |
| `fourier-pom-fla-gdn-dense-tied` | 4.04 | 17,523,784 | spectral body + strict tie |
| `dense-tied-attention` | 6.01 | 18,219,008 | original body + strict tie |

On this run, **projected-dense-tied beats the original untied dense-attention path** on eval while using **~half the parameters**. The earlier Colab Exp 25 row (`dense-attention` **2.47**) used the same script shape but a different token slice / GPU; treat **3.55 vs 3.17** as the paired comparison on current local README-BPE data.

### Real HRM laptop slice (~5M tokens, local CUDA, 2026-05-22)

`--data-source hrm_slice` — `data_io/data_laptop_hrm_slice` (gsm8k + math + no_robots + 25k webinstruct).

| Variant | Mean eval ↓ | Params |
| --- | ---: | ---: |
| **`fourier-pom-fla-gdn-projected-dense-tied`** | **5.68** | 17,654,856 |
| `dense-attention` | 5.84 | 34,996,224 |
| `fourier-pom-fla-gdn-dense-tied` | 5.89 | 17,523,784 |
| `dense-tied-attention` | 6.07 | 18,219,008 |

Real inst/resp text is **harder** than README BPE (~5.7 vs ~3.2), but the **ranking holds**: projected + spectral body still wins vs original untied at ~half vocab params.

**Follow-up:** [Experiment 32 - Laptop HRM Ladder](../Experiment%2032%20-%20Laptop%20HRM%20Ladder/README.md) — **500 steps**, h ∈ {256,384,512}: projected **4.54–4.61** vs untied **4.92–4.94** on the same slice (~44 min local CUDA).

## Open questions

- Can a hybrid Fourier vocab/head close the gap to dense tied without going back to 17M vocab params?
- Does `fourier-pom-fla-gdn-dense-tied` stay strong on a longer run, or is this mostly fast tiny-data fitting?
- At what point does the untied dense vocab/head advantage matter less than model body efficiency?
