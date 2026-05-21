from pathlib import Path
import argparse
import importlib.util
import statistics
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_target_sweep():
    probe_path = REPO_ROOT / "experiments" / "Experiment 5 - Fourier Attention Targets" / "fourier_target_sweep.py"
    spec = importlib.util.spec_from_file_location("fourier_target_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TARGET_SWEEP = _load_target_sweep()
HOLDOUT = TARGET_SWEEP.HOLDOUT


def parse_target_modes(value: str) -> list[tuple[str, int]]:
    result = []
    for group in value.split(";"):
        group = group.strip()
        if not group:
            continue
        target, raw_modes = group.split(":", 1)
        target = target.strip()
        if target not in {"mlp", "attention", "all"}:
            raise ValueError(f"Unknown target: {target}")
        for raw_mode in raw_modes.split(","):
            raw_mode = raw_mode.strip()
            if raw_mode:
                result.append((target, int(raw_mode)))
    if not result:
        raise ValueError("At least one target:mode pair is required.")
    return result


def select_top_configs(rows: list[dict[str, float | int | str]], *, top_k: int) -> list[tuple[str, int]]:
    best_by_config: dict[tuple[str, int], float] = {}
    for row in rows:
        config = (str(row["target"]), int(row["mode"]))
        score = float(row["final_eval"])
        if config not in best_by_config or score < best_by_config[config]:
            best_by_config[config] = score
    return [config for config, _score in sorted(best_by_config.items(), key=lambda item: item[1])[:top_k]]


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


def print_summary(rows: list[dict[str, float | int | str]], *, title: str) -> None:
    print(title)
    for variant, item in summarize(rows).items():
        print(
            f"{variant}: runs={item['runs']}, "
            f"mean_final_eval={float(item['mean_final_eval']):.4f}, "
            f"stdev_final_eval={float(item['stdev_final_eval']):.4f}, "
            f"mean_params={int(float(item['mean_num_params'])):,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage target x mode sweep with top-k multi-seed rerun.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scan-seed", type=int, default=1)
    parser.add_argument("--rerun-seeds", default="1,2,3")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--target-modes", default="mlp:48,64,96;attention:48,64,96;all:64,96,128")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=8)
    args = parser.parse_args()

    target_modes = parse_target_modes(args.target_modes)
    rerun_seeds = HOLDOUT.parse_int_list(args.rerun_seeds)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + args.eval_batches + 8)
    tokens = HOLDOUT.TEXT_PROBE.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)
    train_tokens, eval_tokens = HOLDOUT.split_train_eval_tokens(tokens, eval_fraction=args.eval_fraction)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, train={train_tokens.numel():,}, eval={eval_tokens.numel():,}, steps={args.steps}")
    print(f"scan_seed={args.scan_seed}, target_modes={target_modes}")

    scan_rows: list[dict[str, float | int | str]] = []
    dense_scan = TARGET_SWEEP.train_once(
        target="dense",
        mode=target_modes[0][1],
        seed=args.scan_seed,
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
    dense_scan["mode"] = target_modes[0][1]
    scan_rows.append(dense_scan)
    print(TARGET_SWEEP.format_row(dense_scan))

    for target, mode in target_modes:
        row = TARGET_SWEEP.train_once(
            target=target,
            mode=mode,
            seed=args.scan_seed,
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
        row["mode"] = mode
        scan_rows.append(row)
        print(TARGET_SWEEP.format_row(row))

    top_configs = select_top_configs([row for row in scan_rows if row["target"] != "dense"], top_k=args.top_k)
    print_summary(scan_rows, title="scan_summary:")
    print(f"top_configs={top_configs}")

    rerun_rows: list[dict[str, float | int | str]] = []
    for seed in rerun_seeds:
        dense = TARGET_SWEEP.train_once(
            target="dense",
            mode=top_configs[0][1],
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
        dense["mode"] = top_configs[0][1]
        rerun_rows.append(dense)
        print(TARGET_SWEEP.format_row(dense))

        for target, mode in top_configs:
            row = TARGET_SWEEP.train_once(
                target=target,
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
            row["mode"] = mode
            rerun_rows.append(row)
            print(TARGET_SWEEP.format_row(row))

    print_summary(rerun_rows, title="rerun_summary:")


if __name__ == "__main__":
    main()
