"""Exp 30: non-Fourier LM heads on PoM L + FLA-GDN H (vocab/embedding sweep)."""

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

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.layers import Attention, FLAGatedDeltaNetAttention, FourierLinear, PoMAttention
from models.lm_head import (
    DenseTiedVocab,
    LMHead,
    ProjectedDenseTiedVocab,
    TieredHotLowRankVocab,
    TiedKroneckerVocab,
    TiedLowRankVocab,
    UntiedDenseVocab,
    UntiedLowRankVocab,
)


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

DEFAULT_VARIANTS = (
    "spectral-dense-tied,"
    "spectral-projected-dense-tied,"
    "spectral-tied-lowrank,"
    "spectral-tiered-hot-lowrank,"
    "spectral-tied-kronecker,"
    "spectral-untied-lowrank,"
    "spectral-untied-dense"
)

VARIANT_SPECS: dict[str, dict] = {
    "spectral-dense-tied": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "dense_tied"},
    },
    "spectral-projected-dense-tied": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "projected_dense_tied", "bias": True},
    },
    "spectral-tied-lowrank": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "tied_lowrank", "bias": True, "use_lowrank_rank": True},
    },
    "spectral-tiered-hot-lowrank": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "tiered_hot_lowrank", "bias": True, "use_lowrank_rank": True, "use_hot_token_count": True},
    },
    "spectral-tied-kronecker": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "tied_kronecker", "bias": True, "use_kronecker_factors": True},
    },
    "spectral-untied-lowrank": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "untied_lowrank", "bias": True, "use_lowrank_rank": True},
    },
    "spectral-untied-dense": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "untied_dense", "bias": True},
    },
}


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if not variants:
        raise ValueError("At least one variant is required.")
    return variants


def make_config(*,
                variant: str,
                vocab_size: int,
                seq_len: int,
                hidden_size: int,
                lowrank_rank: int,
                hot_token_count: int,
                vocab_factor_a: int,
                vocab_factor_b: int,
                hidden_factor_a: int,
                hidden_factor_b: int) -> dict:
    spec = VARIANT_SPECS[variant]
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
    if config["vocab_head"].pop("use_lowrank_rank", None):
        config["vocab_head"]["lowrank_rank"] = lowrank_rank
    if config["vocab_head"].pop("use_hot_token_count", None):
        config["vocab_head"]["hot_token_count"] = hot_token_count
    if config["vocab_head"].pop("use_kronecker_factors", None):
        config["vocab_head"]["vocab_factor_a"] = vocab_factor_a
        config["vocab_head"]["vocab_factor_b"] = vocab_factor_b
        config["vocab_head"]["hidden_factor_a"] = hidden_factor_a
        config["vocab_head"]["hidden_factor_b"] = hidden_factor_b
    config["token_mixer"] = "pom"
    config["H_override"] = dict(config.get("H_override", {})) | {
        "token_mixer": "fla_gdn",
        "fourier_linear": dict(config.get("fourier_linear", {"enabled": False})) | {"enabled": False},
    }
    return config


def count_modules(model: torch.nn.Module) -> dict[str, int]:
    return {
        "attention_modules": sum(1 for module in model.modules() if isinstance(module, Attention)),
        "fourier_modules": sum(1 for module in model.modules() if isinstance(module, FourierLinear)),
        "pom_modules": sum(1 for module in model.modules() if isinstance(module, PoMAttention)),
        "fla_gdn_modules": sum(1 for module in model.modules() if isinstance(module, FLAGatedDeltaNetAttention)),
        "dense_tied_vocab_modules": sum(1 for module in model.modules() if isinstance(module, DenseTiedVocab)),
        "projected_dense_vocab_modules": sum(1 for module in model.modules() if isinstance(module, ProjectedDenseTiedVocab)),
        "tied_lowrank_vocab_modules": sum(1 for module in model.modules() if isinstance(module, TiedLowRankVocab)),
        "tiered_hot_lowrank_vocab_modules": sum(1 for module in model.modules() if isinstance(module, TieredHotLowRankVocab)),
        "tied_kronecker_vocab_modules": sum(1 for module in model.modules() if isinstance(module, TiedKroneckerVocab)),
        "untied_dense_vocab_modules": sum(1 for module in model.modules() if isinstance(module, UntiedDenseVocab)),
        "untied_lowrank_vocab_modules": sum(1 for module in model.modules() if isinstance(module, UntiedLowRankVocab)),
    }


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
               variant: str,
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
               lowrank_rank: int,
               hot_token_count: int,
               vocab_factor_a: int,
               vocab_factor_b: int,
               hidden_factor_a: int,
               hidden_factor_b: int) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    total_len = prefix_len + causal_len
    config = make_config(
        variant=variant,
        vocab_size=vocab_size,
        seq_len=total_len,
        hidden_size=hidden_size,
        lowrank_rank=lowrank_rank,
        hot_token_count=hot_token_count,
        vocab_factor_a=vocab_factor_a,
        vocab_factor_b=vocab_factor_b,
        hidden_factor_a=hidden_factor_a,
        hidden_factor_b=hidden_factor_b,
    )
    model = LMHead(HierarchicalReasoningModel(config), config).to(device)
    for module in model.modules():
        if hasattr(module, "set_token_permutation"):
            module.set_token_permutation(token_permutation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

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

    modules = count_modules(model)
    return {
        "variant": variant,
        "seed": seed,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
        **{key: float(value) for key, value in modules.items()},
    }


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    result = {}
    for variant, variant_rows in grouped.items():
        evals = [float(row["final_eval"]) for row in variant_rows]
        params = [float(row["num_params"]) for row in variant_rows]
        result[variant] = {
            "runs": len(variant_rows),
            "mean_final_eval": sum(evals) / len(evals),
            "stdev_final_eval": statistics.pstdev(evals) if len(evals) > 1 else 0.0,
            "mean_num_params": sum(params) / len(params),
            "mean_peak_vram_mb": sum(float(row["peak_vram_mb"]) for row in variant_rows) / len(variant_rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 30 non-Fourier vocab heads on PoM + FLA-GDN.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--numseqs", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--causal-len", type=int, default=128)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lowrank-rank", type=int, default=128)
    parser.add_argument("--hot-token-count", type=int, default=4096)
    parser.add_argument("--vocab-factor-a", type=int, default=256)
    parser.add_argument("--vocab-factor-b", type=int, default=256)
    parser.add_argument("--hidden-factor-a", type=int, default=16)
    parser.add_argument("--hidden-factor-b", type=int, default=16)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    variants = parse_variants(args.variants)
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
    print(f"lowrank_rank={args.lowrank_rank}, hot_token_count={args.hot_token_count}")
    print(f"variants={variants}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for variant in variants:
            row = train_once(
                variant=variant,
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
                lowrank_rank=args.lowrank_rank,
                hot_token_count=args.hot_token_count,
                vocab_factor_a=args.vocab_factor_a,
                vocab_factor_b=args.vocab_factor_b,
                hidden_factor_a=args.hidden_factor_a,
                hidden_factor_b=args.hidden_factor_b,
            )
            rows.append(row)
            print(
                f"{row['variant']} seed={row['seed']}: "
                f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
                f"params={int(float(row['num_params'])):,}, "
                f"peak_vram_mb={float(row['peak_vram_mb']):.1f}"
            )

    print("summary:")
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_peak_vram_mb={float(item['mean_peak_vram_mb']):.1f}"
        )


if __name__ == "__main__":
    main()
