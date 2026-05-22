# Colab / Kaggle workflow (Experiment 29 follow-up)

**Colab quota exhausted?** Use **[kaggle/KAGGLE.md](../../../kaggle/KAGGLE.md)** (API `kernels push` or web UI, ~30 GPU h/week).

# Colab workflow (Experiment 29 follow-up)

Open **`colab/experiment_29_followup.ipynb`** in Google Colab, set **GPU** runtime, run all cells.

## Before you run

1. **Push** this repo (Exp 29 scripts + new vocab heads) to `NASAEXP/Spectral-HRM`, **or** upload/mount your local clone and `%cd` there instead of `git clone`.
2. Colab clones **`sapientinc/data_io`** for `tokenizer.json` (same as Exp 23/25).

## Triton install (`status=missing_triton`)

`pip install flash-linear-attention` **does not** install Triton. Colab often has `fla` but no `triton` module.

**Fix (run in repo root after clone):**

```python
!python colab/install_fla_colab.py
```

That script picks a FLA-hosted Triton wheel matched to your Colab `torch` version (see [FLA FAQs](https://github.com/fla-org/flash-linear-attention/blob/main/FAQs.md)) and re-runs the Exp 22 probe until `status=ready`.

**Manual fallback** (if the script fails, check `torch.__version__` first):

```python
import torch
print(torch.__version__)
# PyTorch 2.6.x:
!pip install -q triton==3.2.0 --index-url https://pypi.fla-org.com/simple
# PyTorch 2.7.1:
# !pip install -q triton==3.3.1 --index-url https://pypi.fla-org.com/simple
!pip install -q -r requirements-fla.txt
!python "experiments/Experiment 22 - FLA GDN Kernel Probe/fla_gdn_probe.py"
```

If Triton still will not install, run **Phase A only** (attention H-level). Phase B needs `status=ready`.

## Phase A — hidden_modes sweep (~30–60 min)

```python
!python "experiments/Experiment 29 - Vocab Head Pareto Sweep/colab_followup_sweep.py" \
  --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda \
  --tokenizer-path /content/data_io/trained_tokenizers/bpe/tokenizer.json \
  --variants pom-tied-fourier,pom-multiscale-fourier,pom-tiered-hot-fourier \
  --hidden-modes-list 64,128,192,256
```

Pick the best `hidden_modes` per variant from the `summary:` block.

## Phase B — FLA-GDN port (~15–30 min)

Update `HIDDEN_MODES` and `MULTISCALE` in the notebook, then:

```python
!python "experiments/Experiment 29 - Vocab Head Pareto Sweep/fla_gdn_port_sweep.py" \
  --steps 40 --warmup-steps 1 --seeds 1,2,3 --device cuda \
  --tokenizer-path /content/data_io/trained_tokenizers/bpe/tokenizer.json \
  --hidden-modes 128 \
  --multiscale-specs "512,128;256,256"
```

Variants: `spectral-tied-fourier`, `spectral-multiscale-fourier`, `spectral-tiered-hot-fourier`, `spectral-dense-tied`.

## After Colab

Paste both `summary:` sections into Cursor chat to update the README and lock a yaml preset.
