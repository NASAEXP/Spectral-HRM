# Kaggle GPU for Experiment 29

Use Kaggle when Colab quota is exhausted. Same Linux + CUDA path; same `colab/install_fla_colab.py` for Triton.

## Quota

- About **30 GPU hours per week** (check [kaggle.com/settings](https://www.kaggle.com/settings) → GPU quota).
- API can only turn GPU **on/off** (`enable_gpu`); accelerator type (T4 vs P100) is chosen in the web UI or preserved in pulled notebook metadata.

## Option A — Web UI (simplest)

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **Notebook** or **Script**.
2. **Settings** → **Accelerator** → **GPU T4 x2** (or GPU), **Internet** → **On**.
3. Upload / clone repo, or paste cells from `colab/experiment_29_followup.ipynb`.
4. First run:

   ```python
   !git clone --depth 1 https://github.com/NASAEXP/Spectral-HRM.git /kaggle/working/Spectral-HRM
   !git clone --depth 1 https://github.com/sapientinc/data_io.git /kaggle/working/data_io
   %cd /kaggle/working/Spectral-HRM
   !python colab/install_fla_colab.py
   ```

5. Then `colab_followup_sweep.py` and optionally `fla_gdn_port_sweep.py` (see `experiments/Experiment 29 - …/COLAB.md`).

## Option B — Kaggle API (`kaggle kernels push`)

### One-time setup

```powershell
pip install kaggle
```

Create API token: Kaggle → **Account** → **Create New Token** → saves `kaggle.json`.

Windows:

```powershell
mkdir $env:USERPROFILE\.kaggle
move Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

### Push this script kernel

1. Edit `kaggle/kernel-metadata.json`:
   - Set `"id"` to `YOUR_USERNAME/spectral-hrm-exp29` (slug must be unique on your account).
2. From repo root:

```powershell
cd C:\Users\Dos\Documents\GRAM\Spectral-HRM
kaggle kernels push -p kaggle
```

That uploads `experiment_29_followup.py` and **starts a GPU run**. Watch progress:

```powershell
kaggle kernels status YOUR_USERNAME/spectral-hrm-exp29 -v
```

Download logs/output:

```powershell
kaggle kernels output YOUR_USERNAME/spectral-hrm-exp29 -p ./kaggle-output
```

### Push a notebook instead

Change `kernel-metadata.json`:

```json
"code_file": "experiment_29_followup.ipynb",
"language": "python",
"kernel_type": "notebook"
```

(You can export the Colab notebook into `kaggle/`.)

## Option C — Push your local branch (unpublished Exp 29)

The script clones `NASAEXP/Spectral-HRM` from GitHub. If Exp 29 is **not pushed yet**, either:

- **Push** your branch to GitHub first, or
- Switch `experiment_29_followup.py` to copy from `/kaggle/input/...` if you upload a Dataset zip of your repo.

## API limits (what it is / isn’t)

| Works | Doesn’t |
| --- | --- |
| `kernels push` + GPU via `enable_gpu: true` | “Run this existing notebook again” without re-push |
| Batch runs from your machine / CI | Pick T4 vs P100 reliably via API (use UI once, then `kernels pull -m`) |
| `kernels output` to fetch artifacts | Real-time streaming like Colab (batch only) |

Newer CLI builds add `--accelerator` / `--acc` (see [kaggle-cli#821](https://github.com/Kaggle/kaggle-cli/issues/821)); if your `kaggle` is old, upgrade: `pip install -U kaggle`.

## Triton on Kaggle

Same error as Colab (`status=missing_triton`) → run:

```python
!python colab/install_fla_colab.py
```

If it still fails, Phase A (`colab_followup_sweep.py`) runs without FLA; set `RUN_FLA_PORT = False` in `experiment_29_followup.py`.

## After the run

Paste the `summary:` blocks here (or attach `kaggle-output`) so we can update Exp 29 README and lock presets.
