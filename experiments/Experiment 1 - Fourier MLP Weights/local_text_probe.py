from pathlib import Path
import argparse
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.common import IGNORE_LABEL_ID
from models.layers import FourierLinear
from models.lm_head import LMHead


def byte_tokenize(text: str, vocab_size: int) -> torch.Tensor:
    if vocab_size < 259:
        raise ValueError("vocab_size must be at least 259 for byte tokenization.")
    data = list(text.encode("utf-8"))
    return torch.tensor([(byte % (vocab_size - 1)) + 1 for byte in data], dtype=torch.long)


def load_local_text_tokens(vocab_size: int, min_tokens: int) -> torch.Tensor:
    parts = []
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "dataset_new.py", REPO_ROOT / "models" / "layers.py"):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))

    text = "\n\n".join(parts)
    tokens = byte_tokenize(text, vocab_size=vocab_size)
    if tokens.numel() < min_tokens + 1:
        repeats = (min_tokens + 1 + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    return tokens


def make_config(*, fourier: bool, vocab_size: int, seq_len: int, hidden_size: int, modes: int) -> dict:
    config = {
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "n_layers": 2,
        "hidden_size": hidden_size,
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
            "in_modes": modes,
            "out_modes": modes,
        }
    return config


def make_prefixlm_batch(tokens: torch.Tensor,
                        *,
                        offset: int,
                        numseqs: int,
                        prefix_len: int,
                        causal_len: int,
                        device: torch.device) -> dict[str, torch.Tensor]:
    total_len = prefix_len + causal_len
    inputs, labels, position_ids = [], [], []
    prefix_lens, causal_lens, cu_seqlens = [], [], [0]

    for seq_idx in range(numseqs):
        start = (offset + seq_idx * total_len) % (tokens.numel() - total_len - 1)
        full = tokens[start:start + total_len + 1]
        seq_inputs = full[:-1].clone()
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


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module,
                  tokens: torch.Tensor,
                  *,
                  device: torch.device,
                  batches: int,
                  numseqs: int,
                  prefix_len: int,
                  causal_len: int) -> float:
    model.eval()
    losses = []
    total_len = prefix_len + causal_len
    for idx in range(batches):
        batch = make_prefixlm_batch(tokens, offset=idx * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train_variant(name: str,
                  *,
                  fourier: bool,
                  tokens: torch.Tensor,
                  steps: int,
                  device: torch.device,
                  hidden_size: int,
                  modes: int,
                  numseqs: int,
                  prefix_len: int,
                  causal_len: int) -> dict[str, float]:
    torch.manual_seed(11)
    vocab_size = 260
    total_len = prefix_len + causal_len
    model = LMHead(
        HierarchicalReasoningModel(make_config(fourier=fourier, vocab_size=vocab_size, seq_len=total_len, hidden_size=hidden_size, modes=modes)),
        {"vocab_size": vocab_size},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    first_eval = evaluate_loss(model, tokens, device=device, batches=4, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len)
    train_loss = first_eval
    for step in range(steps):
        batch = make_prefixlm_batch(tokens, offset=step * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

    final_eval = evaluate_loss(model, tokens, device=device, batches=4, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    elapsed = time.perf_counter() - start
    return {
        "first_eval": first_eval,
        "train_loss": train_loss,
        "final_eval": final_eval,
        "elapsed_s": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "fourier_modules": float(count_fourier_modules(model)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local README byte-text dense-vs-Fourier probe.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + 8)
    tokens = load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, steps={args.steps}, hidden_size={args.hidden_size}, modes={args.modes}")
    for name, fourier in (("dense", False), ("fourier", True)):
        result = train_variant(
            name,
            fourier=fourier,
            tokens=tokens,
            steps=args.steps,
            device=device,
            hidden_size=args.hidden_size,
            modes=args.modes,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
        )
        print(
            f"{name}: "
            f"eval {result['first_eval']:.4f} -> {result['final_eval']:.4f}, "
            f"last_train_loss={result['train_loss']:.4f}, "
            f"params={int(result['num_params']):,}, "
            f"fourier_modules={int(result['fourier_modules'])}, "
            f"peak_vram_mb={result['peak_vram_mb']:.1f}, "
            f"elapsed_s={result['elapsed_s']:.2f}"
        )


if __name__ == "__main__":
    main()
