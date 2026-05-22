"""Hidden-modes Pareto follow-up (Exp 29 winners). Run on Colab with GPU."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _import_sweep():
    sweep_path = REPO_ROOT / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "vocab_pareto_sweep.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("vocab_pareto_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_hidden_modes_list(value: str) -> list[int]:
    modes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one hidden mode is required.")
    return modes


def multiscale_specs_for_hidden_modes(*, hidden_modes: int, hidden_size: int, vocab_modes: int) -> list[tuple[int, int]]:
    second_hidden = min(hidden_size, max(hidden_modes, hidden_modes * 2))
    second_vocab = max(hidden_modes, vocab_modes // 2)
    return [(vocab_modes, hidden_modes), (second_vocab, second_hidden)]


def run_hidden_modes_sweep(args: argparse.Namespace) -> None:
    sweep = _import_sweep()
    hidden_modes_list = parse_hidden_modes_list(args.hidden_modes_list)
    variants = sweep.parse_variants(args.variants)
    seeds = sweep.HOLDOUT.parse_int_list(args.seeds)

    if args.device == "auto":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch

        device = torch.device(args.device)

    tokenizer_path = args.tokenizer_path
    id_to_token = sweep.TOKENIZER_ORDERING.load_id_to_token(tokenizer_path)
    vocab_size = len(id_to_token)
    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.warmup_steps + args.eval_batches + 8)
    tokens = sweep.TOKENIZER_ORDERING.load_tokenizer_tokens(tokenizer_path, min_tokens=tokens_needed)
    train_tokens, eval_tokens = sweep.HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)
    token_permutation = sweep.TOKENIZER_ORDERING.make_token_permutation(
        "token_frequency",
        tokens=train_tokens,
        id_to_token=id_to_token,
    )

    print(f"device={device}")
    print(f"hidden_modes_list={hidden_modes_list}")
    print(f"variants={variants}")

    rows: list[dict[str, float | int | str]] = []
    for hidden_modes in hidden_modes_list:
        multiscale_specs = multiscale_specs_for_hidden_modes(
            hidden_modes=hidden_modes,
            hidden_size=args.hidden_size,
            vocab_modes=args.vocab_modes,
        )
        print(f"--- hidden_modes={hidden_modes}, multiscale_specs={multiscale_specs} ---")
        for seed in seeds:
            for variant in variants:
                label = f"{variant}@hm{hidden_modes}"
                row = sweep.train_once(
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
                    hidden_modes=hidden_modes,
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
                row["label"] = label
                row["hidden_modes"] = float(hidden_modes)
                rows.append(row)
                print(f"{label} seed={row['seed']}: eval -> {float(row['final_eval']):.4f}, params={int(float(row['num_params'])):,}")

    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    print("summary:")
    for label, label_rows in grouped.items():
        evals = [float(item["final_eval"]) for item in label_rows]
        params = [float(item["num_params"]) for item in label_rows]
        mean_eval = sum(evals) / len(evals)
        mean_params = sum(params) / len(params)
        print(f"{label}: mean_final_eval={mean_eval:.4f}, mean_params={int(mean_params):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 29 Colab follow-up: hidden_modes sweep for winning vocab heads.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--variants",
        default="pom-tied-fourier,pom-multiscale-fourier,pom-tiered-hot-fourier",
    )
    parser.add_argument("--hidden-modes-list", default="64,128,192,256")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--numseqs", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--causal-len", type=int, default=128)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--fourier-mode", type=int, default=64)
    parser.add_argument("--pom-order", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--hot-token-count", type=int, default=4096)
    parser.add_argument("--num-clusters", type=int, default=256)
    args = parser.parse_args()

    sweep = _import_sweep()
    if args.tokenizer_path is None:
        args.tokenizer_path = sweep.TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH

    run_hidden_modes_sweep(args)


if __name__ == "__main__":
    main()
