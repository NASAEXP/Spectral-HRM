from pathlib import Path
import argparse
import importlib.util
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_bridge_sweep():
    probe_path = REPO_ROOT / "experiments" / "Experiment 26 - Hybrid Fourier Vocab Bridge" / "vocab_bridge_sweep.py"
    spec = importlib.util.spec_from_file_location("vocab_bridge_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BRIDGE = _load_bridge_sweep()
TOKENIZER_ORDERING = BRIDGE.TOKENIZER_ORDERING
HOLDOUT = BRIDGE.HOLDOUT

DEFAULT_VARIANTS = "dense-tied-attention,spectral-tied-fourier,spectral-hybrid-r128-s025,spectral-hybrid-r128-s050,spectral-hybrid-r128-s100,spectral-hybrid-r128-s200,spectral-dense-tied"
VARIANT_SPECS = {
    "dense-tied-attention": {
        "bridge_variant": "dense-tied-attention",
        "residual_scale": 0.0,
    },
    "spectral-tied-fourier": {
        "bridge_variant": "spectral-tied-fourier",
        "residual_scale": 0.0,
    },
    "spectral-hybrid-r128-s025": {
        "bridge_variant": "spectral-hybrid-r128",
        "residual_scale": 0.25,
    },
    "spectral-hybrid-r128-s050": {
        "bridge_variant": "spectral-hybrid-r128",
        "residual_scale": 0.5,
    },
    "spectral-hybrid-r128-s100": {
        "bridge_variant": "spectral-hybrid-r128",
        "residual_scale": 1.0,
    },
    "spectral-hybrid-r128-s200": {
        "bridge_variant": "spectral-hybrid-r128",
        "residual_scale": 2.0,
    },
    "spectral-dense-tied": {
        "bridge_variant": "spectral-dense-tied",
        "residual_scale": 0.0,
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
    return BRIDGE.make_config(
        variant=spec["bridge_variant"],
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
        pom_order=pom_order,
        residual_scale=spec["residual_scale"],
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
               warmup_steps: int,
               token_permutation: torch.Tensor,
               vocab_size: int) -> dict[str, float | int | str]:
    spec = VARIANT_SPECS[variant]
    row = BRIDGE.train_once(
        variant=spec["bridge_variant"],
        seed=seed,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        steps=steps,
        device=device,
        hidden_size=hidden_size,
        numseqs=numseqs,
        prefix_len=prefix_len,
        causal_len=causal_len,
        eval_batches=eval_batches,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        fourier_mode=fourier_mode,
        pom_order=pom_order,
        residual_scale=spec["residual_scale"],
        warmup_steps=warmup_steps,
        token_permutation=token_permutation,
        vocab_size=vocab_size,
    )
    row["variant"] = variant
    row["bridge_variant"] = spec["bridge_variant"]
    row["residual_scale"] = spec["residual_scale"]
    return row


def format_row(row: dict[str, float | int | str]) -> str:
    return (
        f"{row['variant']} seed={row['seed']} bridge={row['bridge_variant']} scale={float(row['residual_scale']):.2f}: "
        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
        f"last_train_loss={float(row['train_loss']):.4f}, "
        f"params={int(float(row['num_params'])):,}, "
        f"residual_rank={int(float(row['residual_rank']))}, "
        f"hybrid_vocab={int(float(row['hybrid_vocab_modules']))}, "
        f"dense_tied_vocab={int(float(row['dense_tied_vocab_modules']))}, "
        f"tied_fourier_vocab={int(float(row['tied_fourier_vocab_modules']))}, "
        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
        f"elapsed_s={float(row['elapsed_s']):.2f}, "
        f"warmup_elapsed_s={float(row['warmup_elapsed_s']):.2f}, "
        f"train_elapsed_s={float(row['train_elapsed_s']):.2f}, "
        f"ms_per_step={float(row['ms_per_step']):.2f}, "
        f"tokens_per_second={float(row['tokens_per_second']):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep residual_scale for rank-128 hybrid Fourier vocab bridge.")
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
    token_permutation = TOKENIZER_ORDERING.make_token_permutation("token_frequency", tokens=train_tokens, id_to_token=id_to_token)

    print(f"device={device}")
    print(f"tokenizer={args.tokenizer_path}")
    print(f"vocab_size={vocab_size:,}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"context={args.prefix_len}x{args.causal_len}, hidden_size={args.hidden_size}, numseqs={args.numseqs}")
    print(f"vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print(f"pom_order={args.pom_order}, ordering=token_frequency, warmup_steps={args.warmup_steps}")
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
                warmup_steps=args.warmup_steps,
                token_permutation=token_permutation,
                vocab_size=vocab_size,
            )
            rows.append(row)
            print(format_row(row))

    print("summary:")
    for variant, item in BRIDGE.summarize(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}, "
            f"mean_peak_vram_mb={float(item['mean_peak_vram_mb']):.1f}, "
            f"mean_warmup_elapsed_s={float(item['mean_warmup_elapsed_s']):.2f}, "
            f"mean_train_elapsed_s={float(item['mean_train_elapsed_s']):.2f}, "
            f"mean_ms_per_step={float(item['mean_ms_per_step']):.2f}, "
            f"mean_tokens_per_second={float(item['mean_tokens_per_second']):.1f}"
        )


if __name__ == "__main__":
    main()
