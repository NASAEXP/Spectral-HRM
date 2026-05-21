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
from models.lm_head import LMHead


def _load_text_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 1 - Fourier MLP Weights" / "local_text_probe.py"
    spec = importlib.util.spec_from_file_location("local_text_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TEXT_PROBE = _load_text_probe()


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one integer is required.")
    return items


def split_train_eval_tokens(tokens: torch.Tensor, eval_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1.")
    split_at = int(tokens.numel() * (1.0 - eval_fraction))
    return tokens[:split_at].contiguous(), tokens[split_at:].contiguous()


def count_fourier_modules(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, FourierLinear))


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
        batch = TEXT_PROBE.make_prefixlm_batch(tokens, offset=idx * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train_once(*,
               variant: str,
               fourier: bool,
               mode: int,
               seed: int,
               train_tokens: torch.Tensor,
               eval_tokens: torch.Tensor,
               steps: int,
               device: torch.device,
               hidden_size: int,
               numseqs: int,
               prefix_len: int,
               causal_len: int,
               eval_batches: int) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    vocab_size = 260
    total_len = prefix_len + causal_len
    model = LMHead(
        HierarchicalReasoningModel(
            TEXT_PROBE.make_config(
                fourier=fourier,
                vocab_size=vocab_size,
                seq_len=total_len,
                hidden_size=hidden_size,
                modes=mode,
            )
        ),
        {"vocab_size": vocab_size},
    ).to(device)
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
        batch = TEXT_PROBE.make_prefixlm_batch(train_tokens, offset=step * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
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

    return {
        "variant": variant,
        "seed": seed,
        "mode": mode,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "fourier_modules": float(count_fourier_modules(model)),
        "peak_vram_mb": peak_vram_mb,
        "elapsed_s": time.perf_counter() - start,
    }


def summarize_results(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summary = {}
    for variant, variant_rows in grouped.items():
        final_evals = [float(row["final_eval"]) for row in variant_rows]
        num_params = [float(row["num_params"]) for row in variant_rows]
        summary[variant] = {
            "runs": len(variant_rows),
            "mean_final_eval": sum(final_evals) / len(final_evals),
            "stdev_final_eval": statistics.pstdev(final_evals) if len(final_evals) > 1 else 0.0,
            "mean_num_params": sum(num_params) / len(num_params),
        }
    return summary


def format_result_row(row: dict[str, float | int | str]) -> str:
    return (
        f"{row['variant']} seed={row['seed']}: "
        f"eval {float(row['first_eval']):.4f} -> {float(row['final_eval']):.4f}, "
        f"last_train_loss={float(row['train_loss']):.4f}, "
        f"params={int(float(row['num_params'])):,}, "
        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
        f"elapsed_s={float(row['elapsed_s']):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local holdout dense-vs-Fourier probe with multiple seeds.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--modes", default="48,64,96")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    modes = parse_int_list(args.modes)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}, modes={modes}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        dense = train_once(
            variant="dense",
            fourier=False,
            mode=modes[0],
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
        )
        rows.append(dense)
        print(format_result_row(dense))

        for mode in modes:
            variant = f"fourier-{mode}"
            row = train_once(
                variant=variant,
                fourier=True,
                mode=mode,
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
            )
            rows.append(row)
            print(format_result_row(row))

    print("summary:")
    for variant, item in summarize_results(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}"
        )


if __name__ == "__main__":
    main()
