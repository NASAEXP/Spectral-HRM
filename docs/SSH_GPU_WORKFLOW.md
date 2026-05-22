# SSH GPU workflow (no notebooks)

Use a **plain Linux GPU VM** over SSH. Cursor opens the repo on the remote host; the agent runs bash there (same as local, but Linux + Triton + FLA).

Notebooks (Colab / Kaggle / Paperspace UI) are optional legacy paths only.

## 1. Rent a Linux GPU with SSH

Pick any provider that gives **root/ubuntu + SSH + NVIDIA driver** (CUDA preinstalled is ideal):

| Provider | Cost | SSH | Notes |
| --- | --- | --- | --- |
| [Vast.ai](https://vast.ai) | ~$0.15–0.40/hr (often **$5–10** signup credit) | Yes | Cheapest “real VM” path |
| [RunPod](https://runpod.io) | Similar | Yes | Pod = SSH, templates with PyTorch |
| [Lightning.ai](https://lightning.ai) Studio | Free credits + paid | Yes | [SSH docs](https://lightning.ai/docs/overview/ai-studio/ssh-access) |
| Paperspace **Core** (paid) | $8+/mo + GPU hours | Yes | Not the free notebook tier |

**Image:** Ubuntu 22.04 + CUDA 12.x PyTorch template (RunPod/Vast) if offered.

You need:

- Public IP + port 22 (or custom SSH port)
- Your **public key** in `~/.ssh/authorized_keys` on the VM
- Enough disk for `data_io` tokenizer + repo (~2 GB)

## 2. SSH config on your laptop

`%USERPROFILE%\.ssh\config` (Windows):

```sshconfig
Host spectral-gpu
    HostName YOUR_VM_IP
    Port 22
    User root
    IdentityFile ~/.ssh/id_ed25519
```

Test:

```powershell
ssh spectral-gpu nvidia-smi
```

## 3. Cursor Remote SSH

1. Install extension **Remote - SSH** (built into Cursor).
2. **Remote-SSH: Connect to Host…** → `spectral-gpu`.
3. **Open Folder** → `/root/Spectral-HRM` (after setup below).

The agent then runs commands **on the GPU machine**, not on Windows.

## 4. One-shot setup on the VM

From your laptop (or from Cursor terminal on the remote):

```bash
ssh spectral-gpu 'bash -s' < scripts/remote_gpu_setup.sh
```

Or on the VM after `git clone`:

```bash
cd /root/Spectral-HRM
bash scripts/remote_gpu_setup.sh
```

That script:

- Clones `data_io` (tokenizer)
- Installs Python deps + FLA/Triton (`colab/install_fla_colab.py`)
- Verifies `status=ready` on the FLA probe

## 5. Run experiments (shell only)

```bash
cd /root/Spectral-HRM

# Phase A — hidden_modes (PoM L + attention H)
bash scripts/run_exp29_followup.sh attention

# Phase B — FLA-GDN port (needs Triton ready)
bash scripts/run_exp29_followup.sh fla
```

Logs go to `runs/exp29/`. Copy back or paste `summary:` into chat.

## 6. What stays on your laptop

- Windows **without Hyper-V / WSL** is fine.
- Optional: tiny CPU-only tests via `pytest` before pushing to the VM.
- **FLA on Windows** needs [triton-windows](https://github.com/triton-lang/triton-windows); remote Linux is optional, not required for Triton.

## 7. Tear down

Stop/delete the pod on Vast/RunPod when done so hourly billing stops.
