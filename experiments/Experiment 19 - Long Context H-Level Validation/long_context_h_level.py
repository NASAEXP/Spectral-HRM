from pathlib import Path
import argparse
import importlib.util
import statistics
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_h_level_sla():
    probe_path = REPO_ROOT / "experiments" / "Experiment 18 - H-Level SLA" / "h_level_sla.py"
    spec = importlib.util.spec_from_file_location("h_level_sla", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


H_LEVEL = _load_h_level_sla()
TOKENIZER_ORDERING = H_LEVEL.TOKENIZER_ORDERING
HOLDOUT = H_LEVEL.HOLDOUT

DEFAULT_CONTEXTS = "12x12,24x24,48x48"


def parse_contexts(value: str) -> list[tuple[int, int]]:
    contexts: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Context must use prefixxcausal format, got {item!r}.")
        prefix_text, causal_text = item.split("x", maxsplit=1)
        prefix_len, causal_len = int(prefix_text), int(causal_text)
        if prefix_len <= 0 or causal_len <= 0:
            raise ValueError("Context lengths must be positive.")
        contexts.append((prefix_len, causal_len))
    if not contexts:
        raise ValueError("At least one context is required.")
    return contexts


def summarize_by_context(rows: list[dict[str, float | int | str]]) -> dict[tuple[str, str], dict[str, float | int]]:
    grouped: dict[tuple[str, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["context"]), str(row["variant"])), []).append(row)

    result = {}
    for key, key_rows in grouped.items():
        evals = [float(row["final_eval"]) for row in key_rows]
        params = [float(row["num_params"]) for row in key_rows]
        peak_vram = [float(row["peak_vram_mb"]) for row in key_rows]
        elapsed = [float(row["elapsed_s"]) for row in key_rows]
        train_elapsed = [float(row["train_elapsed_s"]) for row in key_rows]
        ms_per_step = [float(row["ms_per_step"]) for row in key_rows]
        tokens_per_second = [float(row["tokens_per_second"]) for row in key_rows]
        result[key] = {
            "runs": len(key_rows),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep context lengths for PoM-L H-level candidates.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=H_LEVEL.DEFAULT_VARIANTS)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--numseqs", type=int, default=2)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--hidden-modes", type=int, default=64)
    parser.add_argument("--fourier-mode", type=int, default=64)
    parser.add_argument("--pom-order", type=int, default=4)
    parser.add_argument("--spectre-buckets", type=int, default=16)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    variants = H_LEVEL.parse_variants(args.variants)
    contexts = parse_contexts(args.contexts)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    id_to_token = TOKENIZER_ORDERING.load_id_to_token(args.tokenizer_path)
    vocab_size = len(id_to_token)
    max_total_len = max(prefix_len + causal_len for prefix_len, causal_len in contexts)
    tokens_needed = max_total_len * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = TOKENIZER_ORDERING.load_tokenizer_tokens(args.tokenizer_path, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)
    token_permutation = TOKENIZER_ORDERING.make_token_permutation("token_frequency", tokens=train_tokens, id_to_token=id_to_token)

    print(f"device={device}")
    print(f"tokenizer={args.tokenizer_path}")
    print(f"vocab_size={vocab_size:,}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print(f"pom_order={args.pom_order}, spectre_buckets={args.spectre_buckets}, ordering=token_frequency")
    print(f"contexts={contexts}")
    print(f"variants={variants}")

    rows: list[dict[str, float | int | str]] = []
    for prefix_len, causal_len in contexts:
        context = f"{prefix_len}x{causal_len}"
        print(f"context={context}")
        for seed in seeds:
            for variant in variants:
                row = H_LEVEL.train_once(
                    variant=variant,
                    seed=seed,
                    train_tokens=train_tokens,
                    eval_tokens=eval_tokens,
                    steps=args.steps,
                    device=device,
                    hidden_size=args.hidden_size,
                    numseqs=args.numseqs,
                    prefix_len=prefix_len,
                    causal_len=causal_len,
                    eval_batches=args.eval_batches,
                    vocab_modes=args.vocab_modes,
                    hidden_modes=args.hidden_modes,
                    fourier_mode=args.fourier_mode,
                    pom_order=args.pom_order,
                    spectre_buckets=args.spectre_buckets,
                    token_permutation=token_permutation,
                    vocab_size=vocab_size,
                )
                row["context"] = context
                rows.append(row)
                print(f"context={context} {H_LEVEL.format_row(row)}")

    print("summary:")
    for (context, variant), item in summarize_by_context(rows).items():
        print(
            f"{context} {variant}: runs={item['runs']}, "
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
