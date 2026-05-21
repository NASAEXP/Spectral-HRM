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
from models.layers import FourierLinear
from models.lm_head import DenseTiedVocab, LMHead, LearnedTokenFourierVocab, TiedFourierVocab, UntiedFourierVocab


def _load_holdout_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 2 - Local Holdout Probe" / "local_holdout_probe.py"
    spec = importlib.util.spec_from_file_location("local_holdout_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HOLDOUT = _load_holdout_probe()


def parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {
        "dense",
        "dense-tied-vocab",
        "tied-fourier-vocab",
        "tied-fourier-vocab-fft-basis",
        "tied-fourier-vocab-bias",
        "tied-fourier-vocab-bias-fft-basis",
        "tied-fourier-vocab-learned-scale",
        "untied-fourier-vocab",
        "learned-token-fourier-vocab",
        "tied-fourier-vocab-reordered",
        "tied-fourier-vocab-checkpoint",
        "fourier-all",
        "fourier-all-dense-tied-vocab",
        "fourier-all-tied-fourier-vocab",
        "fourier-all-tied-fourier-vocab-fft-basis",
        "fourier-all-tied-fourier-vocab-bias",
        "fourier-all-tied-fourier-vocab-bias-fft-basis",
        "fourier-all-untied-fourier-vocab",
        "fourier-all-learned-token-fourier-vocab",
    }
    unknown = [variant for variant in variants if variant not in allowed]
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
                fourier_mode: int) -> dict:
    config = HOLDOUT.TEXT_PROBE.make_config(
        fourier=False,
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        modes=fourier_mode,
    )

    if variant in {"fourier-all", "fourier-all-tied-fourier-vocab"}:
        config["fourier_linear"] = {
            "enabled": True,
            "target": "all",
            "in_modes": fourier_mode,
            "out_modes": fourier_mode,
        }
    elif variant.startswith("fourier-all-"):
        config["fourier_linear"] = {
            "enabled": True,
            "target": "all",
            "in_modes": fourier_mode,
            "out_modes": fourier_mode,
        }

    vocab_head = None
    if variant in {"dense-tied-vocab", "fourier-all-dense-tied-vocab"}:
        vocab_head = {"type": "dense_tied"}
    elif variant in {"tied-fourier-vocab", "fourier-all-tied-fourier-vocab"}:
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
        }
    elif variant in {"tied-fourier-vocab-fft-basis", "fourier-all-tied-fourier-vocab-fft-basis"}:
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "basis_type": "fft",
        }
    elif variant in {"tied-fourier-vocab-bias", "fourier-all-tied-fourier-vocab-bias"}:
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "bias": True,
        }
    elif variant in {"tied-fourier-vocab-bias-fft-basis", "fourier-all-tied-fourier-vocab-bias-fft-basis"}:
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "basis_type": "fft",
            "bias": True,
        }
    elif variant == "tied-fourier-vocab-learned-scale":
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "embedding_scale": "learned",
        }
    elif variant in {"untied-fourier-vocab", "fourier-all-untied-fourier-vocab"}:
        vocab_head = {
            "type": "untied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
        }
    elif variant in {"learned-token-fourier-vocab", "fourier-all-learned-token-fourier-vocab"}:
        vocab_head = {
            "type": "learned_token_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
        }
    elif variant == "tied-fourier-vocab-reordered":
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "token_order": "reverse",
        }
    elif variant == "tied-fourier-vocab-checkpoint":
        vocab_head = {
            "type": "tied_fourier",
            "vocab_modes": vocab_modes,
            "hidden_modes": hidden_modes,
            "checkpoint_weight": True,
        }

    if vocab_head is not None:
        config["vocab_head"] = vocab_head

    return config


def count_modules(model: torch.nn.Module) -> tuple[int, int]:
    fourier = sum(1 for module in model.modules() if isinstance(module, FourierLinear))
    vocab_heads = sum(1 for module in model.modules() if isinstance(module, (DenseTiedVocab, TiedFourierVocab, UntiedFourierVocab, LearnedTokenFourierVocab)))
    return fourier, vocab_heads


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
               token_permutation: torch.Tensor | None = None,
               vocab_size: int = 260) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    total_len = prefix_len + causal_len
    model = LMHead(
        HierarchicalReasoningModel(
            make_config(
                variant=variant,
                vocab_size=vocab_size,
                seq_len=total_len,
                hidden_size=hidden_size,
                vocab_modes=vocab_modes,
                hidden_modes=hidden_modes,
                fourier_mode=fourier_mode,
            )
        ),
        make_config(
            variant=variant,
            vocab_size=vocab_size,
            seq_len=total_len,
            hidden_size=hidden_size,
            vocab_modes=vocab_modes,
            hidden_modes=hidden_modes,
            fourier_mode=fourier_mode,
        ),
    ).to(device)
    if token_permutation is not None:
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
    for step in range(steps):
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(train_tokens, offset=step * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        optimizer.zero_grad(set_to_none=True)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())

    final_eval = evaluate(model, eval_tokens, device=device, batches=eval_batches, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    fourier_modules, vocab_head_modules = count_modules(model)
    return {
        "variant": variant,
        "seed": seed,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "fourier_modules": float(fourier_modules),
        "tied_vocab_modules": float(vocab_head_modules),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
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
        f"last_train_loss={float(row['train_loss']):.4f}, "
        f"params={int(float(row['num_params'])):,}, "
        f"fourier_modules={int(float(row['fourier_modules']))}, "
        f"tied_vocab_modules={int(float(row['tied_vocab_modules']))}, "
        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
        f"elapsed_s={float(row['elapsed_s']):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local dense vocab vs tied Fourier vocab probe.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--variants", default="dense,tied-fourier-vocab,fourier-all,fourier-all-tied-fourier-vocab")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--numseqs", type=int, default=2)
    parser.add_argument("--prefix-len", type=int, default=12)
    parser.add_argument("--causal-len", type=int, default=12)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--vocab-modes", type=int, default=160)
    parser.add_argument("--hidden-modes", type=int, default=64)
    parser.add_argument("--fourier-mode", type=int, default=32)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    variants = parse_variants(args.variants)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}, variants={variants}")
    print(f"hidden_size={args.hidden_size}, vocab_modes={args.vocab_modes}, hidden_modes={args.hidden_modes}, fourier_mode={args.fourier_mode}")

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
            )
            rows.append(row)
            print(format_row(row))

    print("summary:")
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}"
        )


if __name__ == "__main__":
    main()
