from pathlib import Path
import argparse
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.common import IGNORE_LABEL_ID
from models.lm_head import LMHead
from models.layers import FourierLinear


def make_config(*, fourier: bool, vocab_size: int, seq_len: int) -> dict:
    config = {
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "n_layers": 2,
        "hidden_size": 64,
        "num_heads": 4,
        "expansion": 2,
        "attn_type": "prefixlm",
        "init_type": "lecun_normal",
        "norm_type": "pre",
        "norm_eps": 1e-6,
        "pos_emb_type": "none",
        "half_layers": True,
        "H_cycles": 1,
        "L_cycles": 1,
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 2,
        "bp_max_steps": 2,
        "H_override": {},
    }
    if fourier:
        config["fourier_linear"] = {
            "enabled": True,
            "target": "mlp",
            "in_modes": 16,
            "out_modes": 16,
        }
    return config


def make_batch(*, vocab_size: int, numseqs: int, prefix_len: int, causal_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    inputs, labels, position_ids = [], [], []
    prefix_lens, causal_lens, cu_seqlens = [], [], [0]
    total_len = prefix_len + causal_len

    for seq_idx in range(numseqs):
        base = seq_idx * 17 + 3
        full = ((torch.arange(total_len + 1, dtype=torch.long) + base) % (vocab_size - 1)) + 1
        seq_inputs = full[:-1]
        seq_labels = torch.full((total_len,), IGNORE_LABEL_ID, dtype=torch.long)
        seq_labels[prefix_len - 1:] = full[prefix_len:]

        inputs.append(seq_inputs)
        labels.append(seq_labels)
        position_ids.append(torch.arange(total_len, dtype=torch.long))
        prefix_lens.append(prefix_len)
        causal_lens.append(causal_len)
        cu_seqlens.append(cu_seqlens[-1] + total_len)

    return {
        "inputs": torch.cat(inputs).to(device),
        "labels": torch.cat(labels).to(device),
        "position_ids": torch.cat(position_ids).to(device),
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.int32, device=device),
        "causal_lens": torch.tensor(causal_lens, dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        "total_seqlen": torch.tensor(cu_seqlens[-1], dtype=torch.int64, device=device),
        "numseqs": torch.tensor(numseqs, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(prefix_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(causal_len, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(total_len, dtype=torch.int64, device=device),
    }


def count_fourier_modules(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, FourierLinear))


def train_variant(name: str, *, fourier: bool, steps: int, device: torch.device) -> dict[str, float]:
    torch.manual_seed(7)
    vocab_size = 256
    batch = make_batch(vocab_size=vocab_size, numseqs=4, prefix_len=8, causal_len=8, device=device)
    model = LMHead(HierarchicalReasoningModel(make_config(fourier=fourier, vocab_size=vocab_size, seq_len=16)), {"vocab_size": vocab_size}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    first_loss = None
    final_loss = None
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = final_loss

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    elapsed = time.perf_counter() - start
    num_params = sum(param.numel() for param in model.parameters())
    fourier_modules = count_fourier_modules(model)

    return {
        "first_loss": first_loss or 0.0,
        "final_loss": final_loss or 0.0,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "num_params": float(num_params),
        "fourier_modules": float(fourier_modules),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dense-vs-Fourier HRM-Text smoke run.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device={device}")
    for name, fourier in (("dense", False), ("fourier", True)):
        result = train_variant(name, fourier=fourier, steps=args.steps, device=device)
        print(
            f"{name}: "
            f"loss {result['first_loss']:.4f} -> {result['final_loss']:.4f}, "
            f"params={int(result['num_params']):,}, "
            f"fourier_modules={int(result['fourier_modules'])}, "
            f"peak_vram_mb={result['peak_vram_mb']:.1f}, "
            f"elapsed_s={result['elapsed_s']:.2f}"
        )


if __name__ == "__main__":
    main()
