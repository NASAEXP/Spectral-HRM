from pathlib import Path
import argparse
import importlib.util
import json
import statistics
import sys

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAM_ROOT = REPO_ROOT.parent
DEFAULT_TOKENIZER_PATH = GRAM_ROOT / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json"
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

DEFAULT_ORDERINGS = "bpe_rank,token_frequency,token_category,token_category_frequency,random"
LOCKED_ORDERING = "token_frequency"
LOCKED_CONTROL = "bpe_rank"
LOCKED_ORDERINGS = f"{LOCKED_CONTROL},{LOCKED_ORDERING}"
ALLOWED_ORDERINGS = set(DEFAULT_ORDERINGS.split(","))


def parse_orderings(value: str) -> list[str]:
    orderings = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in orderings if item not in ALLOWED_ORDERINGS]
    if unknown:
        raise ValueError(f"Unknown orderings: {unknown}")
    if not orderings:
        raise ValueError("At least one ordering is required.")
    return orderings


def load_id_to_token(tokenizer_path: Path) -> list[str]:
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    id_to_token = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        id_to_token[int(token_id)] = token
    return id_to_token


def _local_text() -> str:
    parts = []
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "dataset_new.py", REPO_ROOT / "models" / "layers.py"):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def load_tokenizer_tokens(
    tokenizer_path: Path,
    *,
    min_tokens: int,
    slice_dir: Path | None = None,
    prefer_slice: bool = True,
) -> torch.Tensor:
    scripts_path = REPO_ROOT / "scripts" / "load_probe_tokens.py"
    spec = importlib.util.spec_from_file_location("load_probe_tokens", scripts_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tokens, _source = module.load_probe_tokens(
        min_tokens=min_tokens,
        tokenizer_path=tokenizer_path,
        slice_dir=slice_dir,
        prefer_slice=prefer_slice,
    )
    return tokens


def _stripped_token(token: str) -> str:
    return token.lstrip("Ġ▁")


def token_text_category(token: str) -> int:
    if token.startswith("<") and token.endswith(">"):
        return 0

    stripped = _stripped_token(token)
    if not stripped or stripped.isspace():
        return 1
    if stripped.isdigit():
        return 2
    if stripped.isascii() and stripped.isalpha():
        return 3
    if stripped.isascii() and stripped.isalnum():
        return 4
    if stripped.isascii() and all(not char.isalnum() and not char.isspace() for char in stripped):
        return 5
    if stripped.isascii():
        return 6
    return 7


def _token_counts(tokens: torch.Tensor, vocab_size: int) -> list[int]:
    counts = torch.bincount(tokens.to(dtype=torch.long).cpu(), minlength=vocab_size)
    return [int(counts[idx]) for idx in range(vocab_size)]


def make_token_permutation(ordering: str, *, tokens: torch.Tensor, id_to_token: list[str]) -> torch.Tensor:
    vocab_size = len(id_to_token)
    if ordering == "bpe_rank":
        ordered = list(range(vocab_size))
    elif ordering == "token_frequency":
        counts = _token_counts(tokens, vocab_size)
        ordered = sorted(range(vocab_size), key=lambda token_id: (-counts[token_id], token_id))
    elif ordering == "token_category":
        ordered = sorted(range(vocab_size), key=lambda token_id: (token_text_category(id_to_token[token_id]), token_id))
    elif ordering == "token_category_frequency":
        counts = _token_counts(tokens, vocab_size)
        ordered = sorted(range(vocab_size), key=lambda token_id: (token_text_category(id_to_token[token_id]), -counts[token_id], token_id))
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
    parser = argparse.ArgumentParser(description="Run tokenizer-aware Fourier vocab ordering checks.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--orderings", default=DEFAULT_ORDERINGS)
    parser.add_argument("--base-variant", default="fourier-all-tied-fourier-vocab-bias")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
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

    id_to_token = load_id_to_token(args.tokenizer_path)
    vocab_size = len(id_to_token)
    total_len = args.prefix_len + args.causal_len
    tokens_needed = total_len * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = load_tokenizer_tokens(args.tokenizer_path, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"base_variant={args.base_variant}")
    print(f"tokenizer={args.tokenizer_path}")
    print(f"vocab_size={vocab_size:,}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}")
    print(f"hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")
    print(f"orderings={orderings}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for ordering in orderings:
            permutation = make_token_permutation(ordering, tokens=train_tokens, id_to_token=id_to_token)
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
                vocab_size=vocab_size,
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
