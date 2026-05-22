"""Exp 31: dense-tied heads with symmetry-compatible vocab optimizers (PoM L + FLA-GDN H)."""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import importlib.util
import statistics
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.layers import Attention, FLAGatedDeltaNetAttention, FourierLinear, PoMAttention
from models.lm_head import DenseTiedVocab, LMHead, ProjectedDenseTiedVocab
from equivariant_vocab_optimizer import build_training_optimizer


def _import_pareto_sweep():
    sweep_path = REPO_ROOT / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "vocab_pareto_sweep.py"
    spec = importlib.util.spec_from_file_location("vocab_pareto_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _import_full_stack():
    probe_path = REPO_ROOT / "experiments" / "Experiment 25 - Full Stack Comparison" / "full_stack_comparison.py"
    spec = importlib.util.spec_from_file_location("full_stack_comparison", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARETO = _import_pareto_sweep()
FULL_STACK = _import_full_stack()
HOLDOUT = FULL_STACK.HOLDOUT
TOKENIZER_ORDERING = FULL_STACK.TOKENIZER_ORDERING

DEFAULT_HEADS = "spectral-dense-tied,spectral-projected-dense-tied"
DEFAULT_VOCAB_OPTIMIZERS = "adamw,rownorm,rightpolar,rightpolar-e10,hybrid"

HEAD_SPECS: dict[str, dict] = {
    "spectral-dense-tied": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "dense_tied"},
    },
    "spectral-projected-dense-tied": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "projected_dense_tied", "bias": True},
    },
}

VOCAB_OPTIMIZER_SPECS: dict[str, dict] = {
    "adamw": {"vocab_optimizer": "adamw"},
    "rownorm": {"vocab_optimizer": "rownorm"},
    "rightpolar": {"vocab_optimizer": "rightpolar", "polar_num_steps": 5, "polar_every": 1},
    "rightpolar-ns2": {"vocab_optimizer": "rightpolar", "polar_num_steps": 2, "polar_every": 1},
    "rightpolar-e10": {"vocab_optimizer": "rightpolar", "polar_num_steps": 5, "polar_every": 10},
    "hybrid": {"vocab_optimizer": "hybrid", "polar_num_steps": 5, "hybrid_alpha": 1.0},
}


def parse_list(value: str, allowed: dict) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown entries: {unknown}")
    if not items:
        raise ValueError("List must not be empty.")
    return items


def run_label(head: str, vocab_opt: str, lr_vocab: float | None = None) -> str:
    if lr_vocab is None:
        return f"{head}+{vocab_opt}"
    return f"{head}+{vocab_opt}+lr_vocab={lr_vocab:g}"


def make_config(*, head: str, vocab_size: int, seq_len: int, hidden_size: int) -> dict:
    spec = HEAD_SPECS[head]
    config = PARETO.VOCAB_PROBE.make_config(
        variant=spec["base_variant"],
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
    )
    vocab_head = dict(spec["vocab_head"])
    head_type = vocab_head.pop("type")
    config["vocab_head"] = {"type": head_type, "bias": vocab_head.pop("bias", False), **vocab_head}
    config["token_mixer"] = "pom"
    config["H_override"] = dict(config.get("H_override", {})) | {
        "token_mixer": "fla_gdn",
        "fourier_linear": dict(config.get("fourier_linear", {"enabled": False})) | {"enabled": False},
    }
    return config


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             tokens: torch.Tensor,
             *,
             device: torch.device,
             batches: int,
             numseqs: int,
             prefix_len: int,
             causal_len: int) -> float:
    return FULL_STACK.evaluate(
        model,
        tokens,
        device=device,
        batches=batches,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
    )


def train_once(*,
               head: str,
               vocab_opt: str,
               seed: int,
               train_tokens: torch.Tensor,
               eval_tokens: torch.Tensor,
               steps: int,
               device: torch.device,
               hidden_size: int,
               numseqs: int,
               prefix_len: int,
               causal_len: int,
               eval_batches: int,
               warmup_steps: int,
               token_permutation: torch.Tensor,
               vocab_size: int,
               lr: float,
               lr_vocab: float | None) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    total_len = prefix_len + causal_len
    config = make_config(head=head, vocab_size=vocab_size, seq_len=total_len, hidden_size=hidden_size)
    model = LMHead(HierarchicalReasoningModel(config), config).to(device)
    for module in model.modules():
        if hasattr(module, "set_token_permutation"):
            module.set_token_permutation(token_permutation)

    opt_spec = VOCAB_OPTIMIZER_SPECS[vocab_opt]
    optimizer = build_training_optimizer(
        model,
        lr=lr,
        lr_vocab=lr_vocab,
        weight_decay=0.01,
        **opt_spec,
    )
    label = run_label(head, vocab_opt, lr_vocab)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    first_eval = evaluate(
        model,
        eval_tokens,
        device=device,
        batches=eval_batches,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
    )
    last_train_loss = first_eval

    for _ in range(warmup_steps):
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(
            train_tokens,
            offset=_ * numseqs * total_len,
            numseqs=numseqs,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())

    for step in range(steps):
        offset = (warmup_steps + step) * numseqs * total_len
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(
            train_tokens,
            offset=offset,
            numseqs=numseqs,
            prefix_len=prefix_len,
            causal_len=causal_len,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())

    final_eval = evaluate(
        model,
        eval_tokens,
        device=device,
        batches=eval_batches,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    return {
        "label": label,
        "head": head,
        "vocab_optimizer": vocab_opt,
        "lr_vocab": float(lr_vocab if lr_vocab is not None else lr),
        "seed": seed,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
    }


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    result = {}
    for label, label_rows in grouped.items():
        evals = [float(row["final_eval"]) for row in label_rows]
        params = [float(row["num_params"]) for row in label_rows]
        result[label] = {
            "runs": len(label_rows),
            "mean_final_eval": sum(evals) / len(evals),
            "stdev_final_eval": statistics.pstdev(evals) if len(evals) > 1 else 0.0,
            "mean_num_params": sum(params) / len(params),
            "mean_peak_vram_mb": sum(float(row["peak_vram_mb"]) for row in label_rows) / len(label_rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 31 symmetry optimizers on tied vocab heads.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--heads", default=DEFAULT_HEADS)
    parser.add_argument("--vocab-optimizers", default=DEFAULT_VOCAB_OPTIMIZERS)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--numseqs", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--causal-len", type=int, default=128)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-vocab", type=float, default=None)
    parser.add_argument(
        "--lr-vocab-grid",
        default="",
        help="Comma-separated vocab LRs; runs one sweep point per value (body --lr unchanged).",
    )
    args = parser.parse_args()

    lr_vocab_values: list[float | None]
    if args.lr_vocab_grid.strip():
        lr_vocab_values = [float(item.strip()) for item in args.lr_vocab_grid.split(",") if item.strip()]
    elif args.lr_vocab is not None:
        lr_vocab_values = [args.lr_vocab]
    else:
        lr_vocab_values = [None]

    seeds = HOLDOUT.parse_int_list(args.seeds)
    heads = parse_list(args.heads, HEAD_SPECS)
    vocab_optimizers = parse_list(args.vocab_optimizers, VOCAB_OPTIMIZER_SPECS)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    id_to_token = TOKENIZER_ORDERING.load_id_to_token(args.tokenizer_path)
    vocab_size = len(id_to_token)
    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.warmup_steps + args.eval_batches + 8)
    tokens = TOKENIZER_ORDERING.load_tokenizer_tokens(args.tokenizer_path, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)
    token_permutation = TOKENIZER_ORDERING.make_token_permutation(
        "token_frequency",
        tokens=train_tokens,
        id_to_token=id_to_token,
    )

    print(f"device={device}")
    print(f"body=L:pom H:fla_gdn")
    print(f"heads={heads}")
    print(f"vocab_optimizers={vocab_optimizers}")
    print(f"body_lr={args.lr}, lr_vocab_values={lr_vocab_values}")
    print("note: RightPolarGrad uses (hidden x hidden) polar, not full vocab SVD")

    rows: list[dict[str, float | int | str]] = []
    for lr_vocab in lr_vocab_values:
        for seed in seeds:
            for head in heads:
                for vocab_opt in vocab_optimizers:
                    row = train_once(
                        head=head,
                        vocab_opt=vocab_opt,
                        seed=seed,
                        train_tokens=train_tokens,
                        eval_tokens=eval_tokens,
                        steps=args.steps,
                        device=device,
                        hidden_size=args.hidden_size,
                        numseqs=args.numseqs,
                        prefix_len=args.prefix_len,
                        causal_len=args.causal_len,
                        eval_batches=args.eval_batches,
                        warmup_steps=args.warmup_steps,
                        token_permutation=token_permutation,
                        vocab_size=vocab_size,
                        lr=args.lr,
                        lr_vocab=lr_vocab,
                    )
                    rows.append(row)
                    print(
                        f"{row['label']} seed={row['seed']}: "
                        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
                        f"params={int(float(row['num_params'])):,}, "
                        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}"
                    )

    print("summary:")
    for label, item in sorted(summarize(rows).items(), key=lambda kv: float(kv[1]["mean_final_eval"])):
        print(
            f"{label}: mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_peak_vram_mb={float(item['mean_peak_vram_mb']):.1f}"
        )


if __name__ == "__main__":
    main()
