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


def _load_holdout_probe():
    probe_path = REPO_ROOT / "experiments" / "Experiment 2 - Local Holdout Probe" / "local_holdout_probe.py"
    spec = importlib.util.spec_from_file_location("local_holdout_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


HOLDOUT = _load_holdout_probe()


def parse_targets(value: str) -> list[str]:
    targets = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"dense", "mlp", "attention", "all"}
    unknown = [target for target in targets if target not in allowed]
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    return targets


def variant_name(target: str, mode: int) -> str:
    if target == "dense":
        return "dense"
    return f"fourier-{target}-{mode}"


def make_config(*, target: str, vocab_size: int, seq_len: int, hidden_size: int, mode: int) -> dict:
    config = HOLDOUT.TEXT_PROBE.make_config(
        fourier=False,
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        modes=mode,
    )
    if target != "dense":
        config["fourier_linear"] = {
            "enabled": True,
            "target": target,
            "in_modes": mode,
            "out_modes": mode,
        }
    return config


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
        batch = HOLDOUT.TEXT_PROBE.make_prefixlm_batch(tokens, offset=idx * numseqs * total_len, numseqs=numseqs, prefix_len=prefix_len, causal_len=causal_len, device=device)
        _carry, loss, _metrics = model(carry=None, batch=batch, bp_steps=2)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train_once(*,
               target: str,
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
        HierarchicalReasoningModel(make_config(target=target, vocab_size=vocab_size, seq_len=total_len, hidden_size=hidden_size, mode=mode)),
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

    return {
        "variant": variant_name(target, mode),
        "seed": seed,
        "target": target,
        "first_eval": first_eval,
        "final_eval": final_eval,
        "train_loss": last_train_loss,
        "num_params": float(sum(param.numel() for param in model.parameters())),
        "fourier_modules": float(count_fourier_modules(model)),
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
        f"peak_vram_mb={float(row['peak_vram_mb']):.1f}, "
        f"elapsed_s={float(row['elapsed_s']):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep dense/MLP/attention/all Fourier targets.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--targets", default="dense,mlp,attention,all")
    parser.add_argument("--mode", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    seeds = HOLDOUT.parse_int_list(args.seeds)
    targets = parse_targets(args.targets)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}, seeds={seeds}, targets={targets}, mode={args.mode}")

    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for target in targets:
            row = train_once(
                target=target,
                mode=args.mode,
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
