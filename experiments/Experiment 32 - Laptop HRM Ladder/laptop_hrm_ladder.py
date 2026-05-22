"""Exp 32: laptop ladder — long train on real HRM slice, scale hidden, vs original-ish control."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_full_stack():
    path = REPO_ROOT / "experiments" / "Experiment 25 - Full Stack Comparison" / "full_stack_comparison.py"
    spec = importlib.util.spec_from_file_location("full_stack_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FS = _load_full_stack()

DEFAULT_VARIANTS = "dense-attention,fourier-pom-fla-gdn-projected-dense-tied"
DEFAULT_HIDDEN_SIZES = "256,384,512"


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("At least one hidden size is required.")
    return sizes


def summarize(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['variant']}|h={row['hidden_size']}"
        grouped.setdefault(key, []).append(row)

    out = {}
    for key, group in grouped.items():
        evals = [float(r["final_eval"]) for r in group]
        vrams = [float(r["peak_vram_mb"]) for r in group]
        out[key] = {
            "runs": len(group),
            "mean_final_eval": sum(evals) / len(evals),
            "stdev_final_eval": statistics.pstdev(evals) if len(evals) > 1 else 0.0,
            "mean_peak_vram_mb": sum(vrams) / len(vrams),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 32 laptop HRM ladder on real slice.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--hidden-sizes", default=DEFAULT_HIDDEN_SIZES)
    parser.add_argument("--tokenizer-path", type=Path, default=FS.TOKENIZER_ORDERING.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--slice-dir", type=Path, default=FS.TOKENIZER_ORDERING.GRAM_ROOT / "data_io" / "data_laptop_hrm_slice")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--numseqs", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=128)
    parser.add_argument("--causal-len", type=int, default=128)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--vocab-modes", type=int, default=512)
    parser.add_argument("--hidden-modes", type=int, default=0, help="0 = auto min(256, hidden_size)")
    parser.add_argument("--fourier-mode", type=int, default=64)
    parser.add_argument("--pom-order", type=int, default=4)
    parser.add_argument(
        "--save-ckpt-dir",
        type=Path,
        default=None,
        help="If set, save each run under <dir>/<variant>_h<size>_s<seed>/ for probe_benchmark.py",
    )
    parser.add_argument(
        "--save-ckpt-filter",
        default=None,
        help="Comma-separated variant names to save (default: all variants in --variants)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch

        device = torch.device(args.device)

    seeds = FS.HOLDOUT.parse_int_list(args.seeds)
    variants = FS.parse_variants(args.variants)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    id_to_token = FS.TOKENIZER_ORDERING.load_id_to_token(args.tokenizer_path)
    vocab_size = len(id_to_token)
    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.warmup_steps + args.eval_batches + 16)
    tokens = FS.TOKENIZER_ORDERING.load_tokenizer_tokens(
        args.tokenizer_path,
        min_tokens=tokens_needed,
        slice_dir=args.slice_dir,
        prefer_slice=True,
    )
    train_tokens, eval_tokens = FS.HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)
    token_permutation = FS.TOKENIZER_ORDERING.make_token_permutation(
        "token_frequency",
        tokens=train_tokens,
        id_to_token=id_to_token,
    )

    print(f"device={device}")
    print(f"data=hrm_slice:{args.slice_dir}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}")
    print(f"steps={args.steps}, seeds={seeds}, hidden_sizes={hidden_sizes}, variants={variants}")
    save_variants: set[str] | None = None
    if args.save_ckpt_filter:
        save_variants = set(FS.parse_variants(args.save_ckpt_filter))
    if args.save_ckpt_dir is not None:
        print(f"save_ckpt_dir={args.save_ckpt_dir}, save_variants={sorted(save_variants) if save_variants else 'all'}")

    rows: list[dict] = []
    for hidden_size in hidden_sizes:
        hidden_modes = args.hidden_modes or min(256, hidden_size)
        for seed in seeds:
            for variant in variants:
                label = f"{variant}|h={hidden_size}"
                try:
                    save_dir = None
                    if args.save_ckpt_dir is not None and (save_variants is None or variant in save_variants):
                        save_dir = args.save_ckpt_dir
                    row = FS.train_once(
                        variant=variant,
                        seed=seed,
                        train_tokens=train_tokens,
                        eval_tokens=eval_tokens,
                        steps=args.steps,
                        device=device,
                        hidden_size=hidden_size,
                        numseqs=args.numseqs,
                        prefix_len=args.prefix_len,
                        causal_len=args.causal_len,
                        eval_batches=args.eval_batches,
                        vocab_modes=args.vocab_modes,
                        hidden_modes=hidden_modes,
                        fourier_mode=args.fourier_mode,
                        pom_order=args.pom_order,
                        warmup_steps=args.warmup_steps,
                        token_permutation=token_permutation,
                        vocab_size=vocab_size,
                        save_ckpt_dir=save_dir,
                        tokenizer_path=args.tokenizer_path,
                    )
                    row["label"] = label
                    row["hidden_size"] = hidden_size
                    rows.append(row)
                    print(
                        f"{label} seed={seed}: "
                        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
                        f"params={int(float(row['num_params'])):,}, "
                        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
                        f"tokens/s={float(row['tokens_per_second']):.0f}"
                        + (f", ckpt={row['ckpt_path']}" if row.get("ckpt_path") else "")
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
                        print(f"{label} seed={seed}: OOM — skip larger configs or lower numseqs")
                        import torch

                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        break
                    raise

    print("summary (lower eval is better):")
    for key, item in sorted(summarize(rows).items(), key=lambda kv: (kv[0].split("|h=")[0], float(kv[0].split("=")[-1]))):
        print(
            f"{key}: mean_final_eval={item['mean_final_eval']:.4f} "
            f"(stdev={item['stdev_final_eval']:.4f}), mean_peak_vram_mb={item['mean_peak_vram_mb']:.1f}"
        )


if __name__ == "__main__":
    main()
