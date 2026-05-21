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

DEFAULT_VARIANTS = (
    "tied-fourier-vocab-bias,"
    "tied-fourier-vocab-bias-fft-basis,"
    "tied-fourier-vocab-reordered,"
    "tied-fourier-vocab-checkpoint,"
    "fourier-all-tied-fourier-vocab-bias,"
    "fourier-all-tied-fourier-vocab-bias-fft-basis"
)

ASSUMPTIONS = {
    "tied-fourier-vocab-bias": "DCT/cosine tied vocab baseline",
    "tied-fourier-vocab-bias-fft-basis": "real FFT-style tied vocab basis",
    "tied-fourier-vocab-reordered": "token order matters",
    "tied-fourier-vocab-checkpoint": "recompute dense W in backward",
    "fourier-all-tied-fourier-vocab-bias": "compressed body plus DCT/cosine vocab",
    "fourier-all-tied-fourier-vocab-bias-fft-basis": "compressed body plus FFT-style vocab",
}


def parse_variants(value: str) -> list[str]:
    return VOCAB_PROBE.parse_variants(value)


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
    parser = argparse.ArgumentParser(description="Run local Fourier vocab assumption checks.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
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
    variants = parse_variants(args.variants)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print("assumptions:")
    for variant in variants:
        print(f"- {variant}: {ASSUMPTIONS.get(variant, 'custom variant')}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for variant in variants:
            row = VOCAB_PROBE.train_once(
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
            )
            rows.append(row)
            print(VOCAB_PROBE.format_row(row))

    print("summary:")
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_elapsed_s={float(item['mean_elapsed_s']):.2f}"
        )


if __name__ == "__main__":
    main()
