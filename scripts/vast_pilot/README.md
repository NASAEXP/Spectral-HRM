# Vast.ai $10 pilot (realistic route)

One rented **A100 40GB** session (~14 h @ $0.67/hr). Goal: **prove pretrain runs** with **VastS + projected/PoM/FLA**, save **one checkpoint**, then **stop billing**.

Not a paper L run. Not GSM8k on this budget unless you have time left after ckpt download.

**Full laptop prep:** [`PREP.md`](PREP.md) · **Data build:** [`../../data_io/VAST_PILOT.md`](../../data_io/VAST_PILOT.md)

## Before you rent (do at home)

1. **Build ~100M token data** — `cd data_io && python build_laptop_hrm_slice.py --preset vast` → `data_vast_pilot_slice/`
2. **Verify** — `python scripts/vast_pilot/verify_dataset.py ../data_io/data_vast_pilot_slice --batch-size 8192`
3. **Pack** — `.\scripts\vast_pilot\pack_for_upload.ps1` → upload `data_io/vast_pilot_data.tar.gz`
4. **WandB** — account ready, or `WANDB_MODE=offline` for smoke only
5. **Rent filter** — **1× A100 40GB**, **~$0.65–0.70/hr**, verified, CUDA 12+, **≥50 GB disk**

## On the instance (~1.5 h setup)

```bash
# SSH in, then:
export WORK=/workspace
mkdir -p $WORK/data $WORK/GRAM
cd $WORK/GRAM
git clone <your-repo-url> .   # or rsync from laptop

cd Spectral-HRM
pip install -r requirements.txt
pip install -r requirements-fla.txt   # if present; FLA + Triton required for fla_gdn

# Unpack data
tar -xf /workspace/data_upload.tar -C $WORK/data
# Must match config/data/hlm_vast.yaml → path: /workspace/data/sampled

# Smoke: 1 step (OOM check)
python pretrain.py --config-name cfg_pretrain_vast \
  data.path=/workspace/data/sampled \
  global_batch_size=4096 epochs=1

# If OOM: global_batch_size=4096 or 2048
# If headroom: global_batch_size=12288
```

## Main train (burn remaining hours)

```bash
cd $WORK/GRAM/Spectral-HRM
export WANDB_API_KEY=...   # optional

python pretrain.py --config-name cfg_pretrain_vast \
  data.path=/workspace/data/sampled \
  global_batch_size=8192 \
  epochs=1
```

Checkpoint: `checkpoints/Spectral-HRM-Vast-Pilot/<run_name>/fsdp2_epoch_1`

## After training

```bash
# Download to laptop (from local machine)
scp -r user@host:/workspace/GRAM/Spectral-HRM/checkpoints/Spectral-HRM-Vast-Pilot ./vast_ckpt
```

**Destroy the Vast instance** in the UI immediately.

## Eval later (not on Vast clock if broke)

```bash
python -m evaluation.main ckpt_path=/path/to/checkpoint_dir \
  run_only=[GSM8k,MATH] generation_config.batch_size=2
```

Use **official ckpt layout** (`all_config.yaml`, `fsdp2_epoch_*`, `carry_epoch_*`). Probe `runs/exp32/ckpts/` layout is different.

## Config reference

| Piece | File |
| --- | --- |
| Train entry | `config/cfg_pretrain_vast.yaml` |
| Size VastS (8L, h=768) | `config/arch/size/VastS.yaml` |
| Net projected+PoM+FLA | `config/arch/net/hrm_vast_projected.yaml` |
| Data path default | `config/data/hlm_vast.yaml` |

## OOM ladder

Try in order: `global_batch_size` **12288 → 8192 → 4096 → 2048**.

## What success looks like

- Training loss decreases over steps without OOM.
- `fsdp2_epoch_1` exists and downloads cleanly.
- Optional: GSM8k smoke on that ckpt on a machine with enough VRAM.

## What failure looks like

- OOM at 2048 → drop to **6 layers** (`VastS.yaml` `n_layers: 6`) or `hidden_size: 512`.
- FLA import fails → rent Linux GPU with Triton; see Exp 23 Colab notebook deps.
