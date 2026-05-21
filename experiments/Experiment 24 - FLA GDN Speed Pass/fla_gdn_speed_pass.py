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
from models.layers import FLAGatedDeltaNetAttention, GatedDeltaNetAttention, PoMAttention, SLAAttention
from models.lm_head import LMHead, TiedFourierVocab


def _load_vocab_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 8 - Tied Fourier Vocab Matrix" / "tied_fourier_vocab_probe.py"
    spec = importlib.util.spec_from_file_location("tied_fourier_vocab_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_tokenizer_ordering():
    ordering_path = REPO_ROOT / "experiments" / "Experiment 14 - Tokenizer-Aware Vocab Ordering" / "tokenizer_aware_vocab_ordering.py"
    spec = importlib.util.spec_from_file_location("tokenizer_aware_vocab_ordering", ordering_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VOCAB_PROBE = _load_vocab_probe()
TOKENIZER_ORDERING = _load_tokenizer_ordering()
HOLDOUT = VOCAB_PROBE.HOLDOUT

DEFAULT_VARIANTS = "pom-sla,pom-gdn,pom-fla-gdn"
VARIANT_SPECS = {
    "pom-sla": {
        "L_mixer": "pom",
        "H_mixer": "sla",
    },
    "pom-gdn": {
        "L_mixer": "pom",
        "H_mixer": "gdn",
    },
    "pom-fla-gdn": {
        "L_mixer": "pom",
        "H_mixer": "fla_gdn",
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
                vocab_modes: int,
                hidden_modes: int,
                fourier_mode: int,
                pom_order: int) -> dict:
    spec = VARIANT_SPECS[variant]
    config = VOCAB_PROBE.make_config(
        variant="fourier-all-tied-fourier-vocab-bias",
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
    )
    config["token_mixer"] = spec["L_mixer"]
    h_override = dict(config.get("H_override", {})) | {"token_mixer": spec["H_mixer"]}
    if spec["H_mixer"] == "fla_gdn":
        h_override["fourier_linear"] = dict(config["fourier_linear"]) | {"enabled": False}
    config["H_override"] = h_override
    config["pom_order"] = pom_order
    return config


def count_mixers(model: torch.nn.Module) -> dict[str, int]:
    return {
        "pom_modules": sum(1 for module in model.modules() if isinstance(module, PoMAttention)),
        "sla_modules": sum(1 for module in model.modules() if isinstance(module, SLAAttention)),
        "gdn_modules": sum(1 for module in model.modules() if isinstance(module, GatedDeltaNetAttention)),
        "fla_gdn_modules": sum(1 for module in model.modules() if isinstance(module, FLAGatedDeltaNetAttention)),
    }


def compute_speed_metrics(*,
                          train_elapsed_s: float,
                          steps: int,
                          numseqs: int,
                          prefix_len: int,
                          causal_len: int) -> dict[str, float]:
    tokens_per_step = float(numseqs * (prefix_len + causal_len))
    if steps <= 0 or train_elapsed_s <= 0:
        return {
            "train_elapsed_s": train_elapsed_s,
            "ms_per_step": 0.0,
            "steps_per_second": 0.0,
            "tokens_per_second": 0.0,
        }

    return {
        "train_elapsed_s": train_elapsed_s,
        "ms_per_step": (train_elapsed_s * 1000.0) / float(steps),
        "steps_per_second": float(steps) / train_elapsed_s,
        "tokens_per_second": (float(steps) * tokens_per_step) / train_elapsed_s,
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
    model.eval()
    losses = []
    total_len = prefix_len + causal_len
    for idx in range(batches):
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(tokens, offset=idx * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


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
               token_permutation: torch.Tensor,
               vocab_size: int) -> dict[str, float | int | str]:
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
    )
    model = LMHead(HierarchicalReasoningModel(config), config).to(device)
    for module in model.modules():
        if isinstance(module, TiedFourierVocab):
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
    train_start = time.perf_counter()
    for step in range(steps):
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(train_tokens, offset=step * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_elapsed_s = time.perf_counter() - train_start

    final_eval = evaluate(model, eval_tokens, device=device, batches=eval_batches, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    mixers = count_mixers(model)
    spec = VARIANT_SPECS[variant]
    speed_metrics = compute_speed_metrics(
        train_elapsed_s=train_elapsed_s,
        steps=steps,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
    )
    return {
        "variant": variant,
        "seed": seed,
        "L_mixer": spec["L_mixer"],
        "H_mixer": spec["H_mixer"],
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "pom_modules": float(mixers["pom_modules"]),
        "sla_modules": float(mixers["sla_modules"]),
        "gdn_modules": float(mixers["gdn_modules"]),
        "fla_gdn_modules": float(mixers["fla_gdn_modules"]),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
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
        peak_vram = [float(row["peak_vram_mb"]) for row in variant_rows]
        elapsed = [float(row["elapsed_s"]) for row in variant_rows]
        train_elapsed = [float(row["train_elapsed_s"]) for row in variant_rows]
        ms_per_step = [float(row["ms_per_step"]) for row in variant_rows]
        tokens_per_second = [float(row["tokens_per_second"]) for row in variant_rows]
        result[variant] = {
            "runs": len(variant_rows),
            "mean_final_eval": sum(evals) / len(evals),
            "stdev_final_eval": statistics.pstdev(evals) if len(evals) > 1 else 0.0,
            "mean_num_params": sum(params) / len(params),
            "mean_peak_vram_mb": sum(peak_vram) / len(peak_vram),
            "mean_elapsed_s": sum(elapsed) / len(elapsed),
            "mean_train_elapsed_s": sum(train_elapsed) / len(train_elapsed),
            "mean_ms_per_step": sum(ms_per_step) / len(ms_per_step),
            "mean_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second),
        }
    return result


def format_row(row: dict[str, float | int | str]) -> str:
    return (
        f"{row['variant']} seed={row['seed']} L={row['L_mixer']} H={row['H_mixer']}: "
        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
        f"last_train_loss={float(row['train_loss']):.4f}, "
        f"params={int(float(row['num_params'])):,}, "
        f"pom={int(float(row['pom_modules']))}, "
        f"sla={int(float(row['sla_modules']))}, "
        f"gdn={int(float(row['gdn_modules']))}, "
        f"fla_gdn={int(float(row['fla_gdn_modules']))}, "
        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
        f"elapsed_s={float(row['elapsed_s']):.2f}, "
        f"train_elapsed_s={float(row['train_elapsed_s']):.2f}, "
        f"ms_per_step={float(row['ms_per_step']):.2f}, "
        f"tokens_per_second={float(row['tokens_per_second']):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FLA Gated DeltaNet against local GDN and SLA.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--numseqs", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=48)
    parser.add_argument("--causal-len", type=int, default=48)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--hidden-modes", type=int, default=64)
    parser.add_argument("--fourier-mode", type=int, default=64)
    parser.add_argument("--pom-order", type=int, default=4)
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
    tokens_needed = total_len * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = TOKENIZER_ORDERING.load_tokenizer_tokens(args.tokenizer_path, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)
    token_permutation = TOKENIZER_ORDERING.make_token_permutation("token_frequency", tokens=train_tokens, id_to_token=id_to_token)

    print(f"device={device}")
    print(f"tokenizer={args.tokenizer_path}")
    print(f"vocab_size={vocab_size:,}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"context={args.prefix_len}x{args.causal_len}, hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print(f"pom_order={args.pom_order}, ordering=token_frequency")
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
                token_permutation=token_permutation,
                vocab_size=vocab_size,
            )
            rows.append(row)
            print(format_row(row))

    print("summary:")
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_peak_vram_mb={float(item['mean_peak_vram_mb']):.1f}, "
            f"mean_elapsed_s={float(item['mean_elapsed_s']):.2f}, "
            f"mean_train_elapsed_s={float(item['mean_train_elapsed_s']):.2f}, "
            f"mean_ms_per_step={float(item['mean_ms_per_step']):.2f}, "
            f"mean_tokens_per_second={float(item['mean_tokens_per_second']):.1f}"
        )


if __name__ == "__main__":
    main()
