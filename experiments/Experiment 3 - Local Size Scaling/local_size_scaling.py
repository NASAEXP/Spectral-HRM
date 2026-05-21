from pathlib import Path
import argparse
import importlib.util
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_holdout_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 2 - Local Holdout Probe" / "local_holdout_probe.py"
    spec = importlib.util.spec_from_file_location("local_holdout_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HOLDOUT = _load_holdout_probe()


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one integer is required.")
    return items


def format_summary_line(*,
                        hidden_size: int,
                        dense_eval: float,
                        fourier_eval: float,
                        dense_params: float,
                        fourier_params: float) -> str:
    delta = fourier_eval - dense_eval
    param_ratio = 100.0 * fourier_params / dense_params
    return (
        f"hidden={hidden_size}: "
        f"dense_eval={dense_eval:.4f}, "
        f"fourier_eval={fourier_eval:.4f}, "
        f"delta={delta:.4f}, "
        f"param_ratio={param_ratio:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale tiny local HRM hidden size for dense vs Fourier-64.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden-sizes", default="96,128,192,256")
    parser.add_argument("--mode", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    hidden_sizes = parse_int_list(args.hidden_sizes)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seed={args.seed}, mode={args.mode}")

    summaries = []
    for hidden_size in hidden_sizes:
        dense = HOLDOUT.train_once(
            variant=f"dense-h{hidden_size}",
            fourier=False,
            mode=args.mode,
            seed=args.seed,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            steps=args.steps,
            device=device,
            hidden_size=hidden_size,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            eval_batches=args.eval_batches,
        )
        print(HOLDOUT.format_result_row(dense))

        fourier = HOLDOUT.train_once(
            variant=f"fourier-{args.mode}-h{hidden_size}",
            fourier=True,
            mode=args.mode,
            seed=args.seed,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            steps=args.steps,
            device=device,
            hidden_size=hidden_size,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
            eval_batches=args.eval_batches,
        )
        print(HOLDOUT.format_result_row(fourier))

        summaries.append(format_summary_line(
            hidden_size=hidden_size,
            dense_eval=float(dense["final_eval"]),
            fourier_eval=float(fourier["final_eval"]),
            dense_params=float(dense["num_params"]),
            fourier_params=float(fourier["num_params"]),
        ))

    print("summary:")
    for line in summaries:
        print(line)


if __name__ == "__main__":
    main()
