from pathlib import Path
import argparse
import importlib.util
import statistics
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_vocab_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 8 - Tied Fourier Vocab Matrix" / "tied_fourier_vocab_probe.py"
    spec = importlib.util.spec_from_file_location("tied_fourier_vocab_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VOCAB_PROBE = _load_vocab_probe()
HOLDOUT = VOCAB_PROBE.HOLDOUT

DEFAULT_ORDERINGS = "identity,frequency,byte_category,byte_category_frequency,random"
ALLOWED_ORDERINGS = set(DEFAULT_ORDERINGS.split(","))


def parse_orderings(value: str) -> list[str]:
    orderings = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in orderings if item not in ALLOWED_ORDERINGS]
    if unknown:
        raise ValueError(f"Unknown orderings: {unknown}")
    if not orderings:
        raise ValueError("At least one ordering is required.")
    return orderings


def _token_counts(tokens: torch.Tensor, vocab_size: int) -> list[int]:
    counts = torch.bincount(tokens.to(dtype=torch.long).cpu(), minlength=vocab_size)
    return [int(counts[idx]) for idx in range(vocab_size)]


def _byte_category(token_id: int) -> int:
    if token_id == 0:
        return 0

    byte = token_id - 1
    if byte in (9, 10, 13, 32):
        return 1
    if 48 <= byte <= 57:
        return 2
    if 65 <= byte <= 90:
        return 3
    if 97 <= byte <= 122:
        return 4
    if 33 <= byte <= 126:
        return 5
    if byte < 128:
        return 6
    return 7


def make_token_permutation(ordering: str, *, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    if ordering == "identity":
        ordered = list(range(vocab_size))
    elif ordering == "frequency":
        counts = _token_counts(tokens, vocab_size)
        ordered = sorted(range(vocab_size), key=lambda token_id: (-counts[token_id], token_id))
    elif ordering == "byte_category":
        ordered = sorted(range(vocab_size), key=lambda token_id: (_byte_category(token_id), token_id))
    elif ordering == "byte_category_frequency":
        counts = _token_counts(tokens, vocab_size)
        ordered = sorted(range(vocab_size), key=lambda token_id: (_byte_category(token_id), -counts[token_id], token_id))
    elif ordering == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1729)
        return torch.randperm(vocab_size, generator=generator, dtype=torch.long)
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    return torch.tensor(ordered, dtype=torch.long)


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    result = {}
    for variant, variant_rows in grouped.items():
        evals = [float(row["final_eval"]) for row in variant_rows]
        params = [float(row["num_params"]) for row in variant_rows]
        elapsed = [float(row["elapsed_s"]) for row in variant_rows]
        result[variant] = {
            "runs": len(variant_rows),
            "mean_final_eval": sum(evals) / len(evals),
            "stdev_final_eval": statistics.pstdev(evals) if len(evals) > 1 else 0.0,
            "mean_num_params": sum(params) / len(params),
            "mean_elapsed_s": sum(elapsed) / len(elapsed),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Fourier vocab token-ordering checks.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--orderings", default=DEFAULT_ORDERINGS)
    parser.add_argument("--base-variant", default="fourier-all-tied-fourier-vocab-bias")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--numseqs", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=12)
    parser.add_argument("--causal-len", type=int, default=12)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--hidden-modes", type=int, default=64)
    parser.add_argument("--fourier-mode", type=int, default=64)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    orderings = parse_orderings(args.orderings)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vocab_size = 260
    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=vocab_size, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"base_variant={args.base_variant}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print(f"orderings={orderings}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for ordering in orderings:
            permutation = make_token_permutation(ordering, tokens=train_tokens, vocab_size=vocab_size)
            row = VOCAB_PROBE.train_once(
                variant=args.base_variant,
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
                token_permutation=permutation,
            )
            row["variant"] = ordering
            rows.append(row)
            print(VOCAB_PROBE.format_row(row))

    print("summary:")
    for ordering, item in summarize(rows).items():
        print(
            f"{ordering}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_elapsed_s={float(item['mean_elapsed_s']):.2f}"
        )


if __name__ == "__main__":
    main()
