# Vast A100 40GB — prep checklist (do on laptop first)

Goal: arrive on the instance with **data packed**, **repo ready**, **WandB optional** — then **smoke → train → download ckpt → destroy**.

## Phase 1 — Data (~100M tokens)

You already have **5M** at `data_io/data_laptop_hrm_slice/`. For the pilot, build **100M**:

```powershell
cd C:\Users\Dos\Documents\GRAM\data_io
pip install huggingface_hub tokenizers numpy
python build_laptop_hrm_slice.py --preset vast
```

Details: [`data_io/VAST_PILOT.md`](../../../data_io/VAST_PILOT.md)

Verify + step count:

```powershell
cd C:\Users\Dos\Documents\GRAM\Spectral-HRM
python scripts/vast_pilot/verify_dataset.py ..\data_io\data_vast_pilot_slice --batch-size 8192
```

Expect **~12,200 steps** for 100M tokens @ `global_batch_size=8192`, 1 epoch.

Pack:

```powershell
.\scripts\vast_pilot\pack_for_upload.ps1 -DatasetDir data_vast_pilot_slice
```

Output: `data_io/vast_pilot_data.tar.gz` (~150–250 MB compressed).

## Phase 2 — Accounts & rent filters

| Item | Action |
|------|--------|
| **Vast** | Account + payment; filter **1× A100 40GB**, CUDA 12+, **≥50 GB disk**, ~**$0.65–0.70/hr** |
| **WandB** | `wandb login` locally, or plan `WANDB_MODE=offline` on box |
| **HF** | `huggingface-cli login` if dataset download prompts (build step) |
| **Budget** | **~$10** → **~14 h** max; plan **~2 h setup**, **~10–12 h train** |

## Phase 3 — What to upload / clone on the instance

**Upload (SCP / Vast file UI):**

- `data_io/vast_pilot_data.tar.gz` → `/workspace/data_upload.tar`

**Clone on GPU (SSH):**

```bash
cd /workspace
git clone https://github.com/NASAEXP/Spectral-HRM.git
cd Spectral-HRM
pip install -r requirements.txt
pip install -r requirements-fla.txt
```

Dataset lives in **`data_io/` on your laptop** — upload the tarball only (not in git).

## Phase 4 — On-instance layout

```bash
mkdir -p /workspace/data/sampled
tar -xzf /workspace/data_upload.tar -C /workspace/data/sampled
ls /workspace/data/sampled   # metadata.json  tokens.npy  epoch_0/
```

## Phase 5 — Smoke then train

```bash
cd /workspace/Spectral-HRM
bash scripts/vast_pilot/vast_instance_bootstrap.sh
bash scripts/vast_pilot/smoke_pretrain_vast.sh /workspace/data/sampled 4096
# if OK, try 8192:
bash scripts/vast_pilot/smoke_pretrain_vast.sh /workspace/data/sampled 8192

export WANDB_API_KEY=...   # optional
python pretrain.py --config-name cfg_pretrain_vast \
  data.path=/workspace/data/sampled \
  global_batch_size=8192 \
  epochs=1
```

OOM: `4096` → `2048`. Headroom: try `12288` in smoke only.

Checkpoint:

```text
checkpoints/Spectral-HRM-Vast-Pilot/<run_name>/fsdp2_epoch_1
```

## Phase 6 — Download & stop billing

From laptop:

```powershell
scp -r user@<vast-host>:/workspace/Spectral-HRM/checkpoints/Spectral-HRM-Vast-Pilot ./vast_ckpt
```

Then **destroy the instance** in Vast UI.

## Phase 7 — Eval at home (free GPU / later)

Use **FSDP checkpoint layout**, not probe `runs/exp32/ckpts/`:

```powershell
python -m evaluation.main ckpt_path=C:\path\to\fsdp2_epoch_1 run_only=[GSM8k]
```

## Optional — local CUDA smoke (before paying)

If your laptop GPU can run VastS + FLA (often tight on 4 GB):

```powershell
cd C:\Users\Dos\Documents\GRAM\Spectral-HRM
$env:PYTHONUNBUFFERED="1"
python pretrain.py --config-name cfg_pretrain_vast `
  data.path=..\data_io\data_laptop_hrm_slice `
  global_batch_size=2048 epochs=1 log_interval=1 lr_warmup_steps=1
```

Uses existing **5M** slice; confirms imports only — not a substitute for 100M cloud train.

## Quick reference

| Piece | Path |
|-------|------|
| Build data | `data_io/build_laptop_hrm_slice.py --preset vast` |
| Train config | `config/cfg_pretrain_vast.yaml` |
| Rent-day README | `scripts/vast_pilot/README.md` |
