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
from models.layers import Attention, FourierLinear, PoMAttention
from models.lm_head import (
    AsymmetricHybridFourierLowRankVocab,
    ClusterHybridFourierLowRankVocab,
    DenseTiedVocab,
    FactorizedFourierVocab,
    HybridFourierLowRankVocab,
    LMHead,
    MultiScaleFourierVocab,
    TieredHotTokenFourierVocab,
    TiedFourierVocab,
)


def _load_full_stack():
    probe_path = REPO_ROOT / "experiments" / "Experiment 25 - Full Stack Comparison" / "full_stack_comparison.py"
    spec = importlib.util.spec_from_file_location("full_stack_comparison", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FULL_STACK = _load_full_stack()
VOCAB_PROBE = FULL_STACK.VOCAB_PROBE
TOKENIZER_ORDERING = FULL_STACK.TOKENIZER_ORDERING
HOLDOUT = FULL_STACK.HOLDOUT

DEFAULT_VARIANTS = (
    "pom-dense-tied,"
    "pom-tied-fourier,"
    "pom-factorized-fourier,"
    "pom-multiscale-fourier,"
    "pom-tiered-hot-fourier,"
    "pom-hybrid-r8,"
    "pom-hybrid-r16,"
    "pom-hybrid-r32,"
    "pom-asymmetric-hybrid-r16,"
    "pom-cluster-hybrid-r16"
)

VARIANT_SPECS: dict[str, dict] = {
    "pom-dense-tied": {
        "base_variant": "fourier-all-dense-tied-vocab",
        "vocab_head": {"type": "dense_tied"},
    },
    "pom-tied-fourier": {
        "base_variant": "fourier-all-tied-fourier-vocab-bias",
        "vocab_head": {"type": "tied_fourier", "bias": True},
    },
    "pom-factorized-fourier": {
        "base_variant": "fourier-all-tied-fourier-vocab-bias",
        "vocab_head": {"type": "factorized_fourier", "bias": True},
    },
    "pom-multiscale-fourier": {
        "base_variant": "fourier-all-tied-fourier-vocab-bias",
        "vocab_head": {"type": "multiscale_fourier", "bias": True, "use_multiscale_specs": True},
    },
    "pom-tiered-hot-fourier": {
        "base_variant": "fourier-all-tied-fourier-vocab-bias",
        "vocab_head": {"type": "tiered_hot_fourier", "bias": True, "use_hot_token_count": True},
    },
    "pom-hybrid-r8": {
        "base_variant": "fourier-all",
        "vocab_head": {"type": "hybrid_fourier_lowrank", "residual_rank": 8, "bias": True},
    },
    "pom-hybrid-r16": {
        "base_variant": "fourier-all",
        "vocab_head": {"type": "hybrid_fourier_lowrank", "residual_rank": 16, "bias": True},
    },
    "pom-hybrid-r32": {
        "base_variant": "fourier-all",
        "vocab_head": {"type": "hybrid_fourier_lowrank", "residual_rank": 32, "bias": True},
    },
    "pom-asymmetric-hybrid-r16": {
        "base_variant": "fourier-all",
        "vocab_head": {"type": "hybrid_fourier_lowrank_asymmetric", "residual_rank": 16, "bias": True},
    },
    "pom-cluster-hybrid-r16": {
        "base_variant": "fourier-all",
        "vocab_head": {"type": "cluster_hybrid_fourier_lowrank", "residual_rank": 16, "bias": True},
    },
}


def parse_multiscale_specs(value: str) -> list[tuple[int, int]]:
    if not value.strip():
        return []
    specs: list[tuple[int, int]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vocab_modes, hidden_modes = chunk.split(",")
        specs.append((int(vocab_modes), int(hidden_modes)))
    return specs


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
                vocab_modes: int,
                hidden_modes: int,
                fourier_mode: int,
                pom_order: int,
                residual_scale: float,
                multiscale_specs: list[tuple[int, int]],
                hot_token_count: int,
                num_clusters: int) -> dict:
    spec = VARIANT_SPECS[variant]
    config = VOCAB_PROBE.make_config(
        variant=spec["base_variant"],
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
    )

    vocab_head = dict(spec["vocab_head"])
    head_type = vocab_head.pop("type")
    config["vocab_head"] = {
        "type": head_type,
        "vocab_modes": vocab_modes,
        "hidden_modes": hidden_modes,
        "bias": vocab_head.pop("bias", False),
        **vocab_head,
    }
    if config["vocab_head"].get("use_multiscale_specs"):
        config["vocab_head"].pop("use_multiscale_specs", None)
        config["vocab_head"]["multiscale_specs"] = multiscale_specs or [(vocab_modes, hidden_modes)]
    if config["vocab_head"].get("use_hot_token_count"):
        config["vocab_head"].pop("use_hot_token_count", None)
        config["vocab_head"]["hot_token_count"] = hot_token_count
    if head_type == "cluster_hybrid_fourier_lowrank":
        config["vocab_head"]["num_clusters"] = num_clusters
    if "residual_rank" in config["vocab_head"]:
        config["vocab_head"]["residual_scale"] = residual_scale

    config["token_mixer"] = "pom"
    config["H_override"] = dict(config.get("H_override", {})) | {
        "token_mixer": "attention",
        "fourier_linear": dict(config.get("fourier_linear", {"enabled": False})) | {"enabled": False},
    }
    config["pom_order"] = pom_order
    return config


def count_modules(model: torch.nn.Module) -> dict[str, int]:
    return {
        "attention_modules": sum(1 for module in model.modules() if isinstance(module, Attention)),
        "fourier_modules": sum(1 for module in model.modules() if isinstance(module, FourierLinear)),
        "pom_modules": sum(1 for module in model.modules() if isinstance(module, PoMAttention)),
        "fla_gdn_modules": 0,
        "dense_tied_vocab_modules": sum(1 for module in model.modules() if isinstance(module, DenseTiedVocab)),
        "tied_fourier_vocab_modules": sum(
            1
            for module in model.modules()
            if isinstance(module, TiedFourierVocab)
            and not isinstance(
                module,
                (
                    HybridFourierLowRankVocab,
                    AsymmetricHybridFourierLowRankVocab,
                    ClusterHybridFourierLowRankVocab,
                    FactorizedFourierVocab,
                    MultiScaleFourierVocab,
                    TieredHotTokenFourierVocab,
                ),
            )
        ),
        "factorized_vocab_modules": sum(1 for module in model.modules() if isinstance(module, FactorizedFourierVocab)),
        "multiscale_vocab_modules": sum(1 for module in model.modules() if isinstance(module, MultiScaleFourierVocab)),
        "hybrid_vocab_modules": sum(
            1
            for module in model.modules()
            if isinstance(module, (HybridFourierLowRankVocab, AsymmetricHybridFourierLowRankVocab, ClusterHybridFourierLowRankVocab))
        ),
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
    first_eval = evaluate(model, eval_tokens, device=device, batches=eval_batches, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len)
    last_train_loss = first_eval

    warmup_start = time.perf_counter()
    for _warmup_step in range(warmup_steps):
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(
            train_tokens,
            offset=_warmup_step * numseqs * total_len,
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    warmup_elapsed_s = time.perf_counter() - warmup_start

    train_start = time.perf_counter()
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_elapsed_s = time.perf_counter() - train_start

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
    speed_metrics = FULL_STACK.compute_speed_metrics(
        train_elapsed_s=train_elapsed_s,
        warmup_elapsed_s=warmup_elapsed_s,
        steps=steps,
        warmup_steps=warmup_steps,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
    )
    spec = VARIANT_SPECS[variant]
    return {
        "variant": variant,
        "seed": seed,
        "base_variant": spec["base_variant"],
        "L_mixer": "pom",
        "H_mixer": "attention",
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
        **{key: float(value) for key, value in modules.items()},
        **speed_metrics,
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
        }
    return result


def format_row(row: dict[str, float | int | str]) -> str:
    return (
        f"{row['variant']} seed={row['seed']}: "
        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
        f"params={int(float(row['num_params'])):,}, "
        f"hybrid_vocab={int(float(row['hybrid_vocab_modules']))}, "
        f"factorized_vocab={int(float(row['factorized_vocab_modules']))}, "
        f"multiscale_vocab={int(float(row['multiscale_vocab_modules']))}, "
        f"tiered_hot_vocab={int(float(row['tiered_hot_vocab_modules']))}, "
        f"fla_gdn={int(float(row['fla_gdn_modules']))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto sweep for new vocab head designs (PoM + attention body, no FLA-GDN).")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    multiscale_specs = parse_multiscale_specs(args.multiscale_specs)
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
    print(f"body=L:pom H:attention (no FLA-GDN)")
    print(f"vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, multiscale_specs={multiscale_specs}")
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
            print(format_row(row))

    print("summary:")
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}"
        )


if __name__ == "__main__":
    main()
