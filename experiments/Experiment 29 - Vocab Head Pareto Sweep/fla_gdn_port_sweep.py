"""Port Exp 29 winning vocab heads onto PoM + FLA-GDN body (Colab / Linux + Triton)."""

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
    HybridFourierLowRankVocab,
    LMHead,
    MultiScaleFourierVocab,
    TieredHotTokenFourierVocab,
    TiedFourierVocab,
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

DEFAULT_VARIANTS = "spectral-tied-fourier,spectral-multiscale-fourier,spectral-tiered-hot-fourier,spectral-dense-tied"

POM_BY_SPECTRAL = {
    "spectral-tied-fourier": "pom-tied-fourier",
    "spectral-multiscale-fourier": "pom-multiscale-fourier",
    "spectral-tiered-hot-fourier": "pom-tiered-hot-fourier",
    "spectral-dense-tied": "pom-dense-tied",
}


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in POM_BY_SPECTRAL]
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
                vocab_modes: int,
                hidden_modes: int,
                fourier_mode: int,
                pom_order: int,
                residual_scale: float,
                multiscale_specs: list[tuple[int, int]],
                hot_token_count: int,
                num_clusters: int) -> dict:
    pom_variant = POM_BY_SPECTRAL[variant]
    config = PARETO.make_config(
        variant=pom_variant,
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
        pom_order=pom_order,
        residual_scale=residual_scale,
        multiscale_specs=multiscale_specs,
        hot_token_count=hot_token_count,
        num_clusters=num_clusters,
    )
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
        "tied_fourier_vocab_modules": sum(
            1
            for module in model.modules()
            if isinstance(module, TiedFourierVocab)
            and not isinstance(module, (HybridFourierLowRankVocab, MultiScaleFourierVocab, TieredHotTokenFourierVocab))
        ),
        "multiscale_vocab_modules": sum(1 for module in model.modules() if isinstance(module, MultiScaleFourierVocab)),
        "tiered_hot_vocab_modules": sum(1 for module in model.modules() if isinstance(module, TieredHotTokenFourierVocab)),
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
               vocab_modes: int,
               hidden_modes: int,
               fourier_mode: int,
               pom_order: int,
               residual_scale: float,
               warmup_steps: int,
               token_permutation: torch.Tensor,
               vocab_size: int,
               multiscale_specs: list[tuple[int, int]],
               hot_token_count: int,
               num_clusters: int) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    total_len = prefix_len + causal_len
    config = make_config(
        variant=variant,
        vocab_size=vocab_size,
        seq_len=total_len,
        hidden_size=hidden_size,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
        pom_order=pom_order,
        residual_scale=residual_scale,
        multiscale_specs=multiscale_specs,
        hot_token_count=hot_token_count,
        num_clusters=num_clusters,
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
    parser = argparse.ArgumentParser(description="FLA-GDN port of Exp 29 vocab head winners.")
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
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--hidden-modes", type=int, default=64)
    parser.add_argument("--fourier-mode", type=int, default=64)
    parser.add_argument("--pom-order", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--multiscale-specs", default="512,64;256,128")
    parser.add_argument("--hot-token-count", type=int, default=4096)
    parser.add_argument("--num-clusters", type=int, default=256)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    variants = parse_variants(args.variants)
    multiscale_specs = PARETO.parse_multiscale_specs(args.multiscale_specs)
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
    token_permutation = TOKENIZER_ORDERING.make_token_permutation("token_frequency", tokens=train_tokens, id_to_token=id_to_token)

    print(f"device={device}")
    print(f"body=L:pom H:fla_gdn")
    print(f"hidden_modes={args.hidden_modes}, multiscale_specs={multiscale_specs}")
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
                vocab_modes=args.vocab_modes,
                hidden_modes=args.hidden_modes,
                fourier_mode=args.fourier_mode,
                pom_order=args.pom_order,
                residual_scale=args.residual_scale,
                warmup_steps=args.warmup_steps,
                token_permutation=token_permutation,
                vocab_size=vocab_size,
                multiscale_specs=multiscale_specs,
                hot_token_count=args.hot_token_count,
                num_clusters=args.num_clusters,
            )
            rows.append(row)
            print(
                f"{row['variant']} seed={row['seed']}: "
                f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
                f"params={int(float(row['num_params'])):,}, "
                f"fla_gdn={int(float(row['fla_gdn_modules']))}, "
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
