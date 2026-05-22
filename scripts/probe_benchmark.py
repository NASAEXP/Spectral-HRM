"""Run GSM8k / MATH (and other evaluation benchmarks) on a saved probe checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.functions import load_model_class

import importlib.util

def _load_probe_generate_nocache():
    path = REPO_ROOT / "scripts" / "probe_generate_nocache.py"
    spec = importlib.util.spec_from_file_location("probe_generate_nocache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.probe_generate_nocache


def _load_probe_checkpoint():
    path = REPO_ROOT / "scripts" / "probe_checkpoint.py"
    spec = importlib.util.spec_from_file_location("probe_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_benchmarks(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one benchmark name is required.")
    return names


def instantiate_benchmark(name: str, *, limit: int | None):
    bench_cls = load_model_class(f"benchmarks@{name}", prefix="evaluation.")
    benchmark = bench_cls()
    if limit is not None and limit > 0:
        benchmark.prompts = benchmark.prompts[:limit]
        benchmark.ground_truths = benchmark.ground_truths[:limit]
    return benchmark


def _score_generation(benchmark, idx: int, text: str) -> dict:
    row: dict = {"parsed": None, "parseable": False, "correct": False}
    if hasattr(benchmark, "_extract_answer"):
        parsed = benchmark._extract_answer(text)  # pyright: ignore[reportAttributeAccessIssue]
        row["parsed"] = parsed
        row["parseable"] = parsed is not None
        if parsed is not None and idx < len(benchmark.ground_truths):
            row["correct"] = parsed == benchmark.ground_truths[idx]
    return row


def _write_sample_records(
    path: Path,
    *,
    ckpt_dir: Path,
    benchmark_name: str,
    benchmark,
    generations: list[str],
    dump_samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe_meta_path = ckpt_dir / "probe_meta.json"
    probe_meta = json.loads(probe_meta_path.read_text(encoding="utf-8")) if probe_meta_path.is_file() else {}

    records = []
    for idx in range(min(dump_samples, len(generations))):
        scores = _score_generation(benchmark, idx, generations[idx])
        records.append({
            "idx": idx,
            "variant": probe_meta.get("variant", ckpt_dir.name),
            "seed": probe_meta.get("seed"),
            "hidden_size": probe_meta.get("hidden_size"),
            "benchmark": benchmark_name,
            "question": benchmark.prompts[idx],
            "ground_truth": benchmark.ground_truths[idx] if idx < len(benchmark.ground_truths) else None,
            "generation": generations[idx],
            **scores,
        })

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_benchmarks(
    ckpt_dir: Path,
    *,
    benchmarks: list[str],
    device: str,
    batch_size: int,
    max_context: int,
    max_tokens: int | None,
    temperature: float,
    condition: str,
    limit: int | None,
    dump_samples: int = 0,
    samples_out: Path | None = None,
    chunk_pad: int = 0,
) -> dict[str, dict]:
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("Probe generation benchmarks require CUDA (inference_generate uses GPU caches).")

    if temperature > 1e-5:
        raise ValueError("Probe benchmarks only support temperature=0 (greedy).")

    ckpt_mod = _load_probe_checkpoint()
    probe_generate = _load_probe_generate_nocache()
    ckpt = ckpt_mod.load_probe_inference_checkpoint(ckpt_dir, device=torch_device)
    if max_tokens is None:
        max_tokens = max_context

    meta_path = Path(ckpt_dir) / "model_config.json"
    model_max = int(json.loads(meta_path.read_text(encoding="utf-8"))["max_seq_len"])
    effective_max = min(max_context, model_max)
    if effective_max < max_context:
        print(f"Note: clamping max_context {max_context} -> {effective_max} (probe max_seq_len={model_max})")
    if chunk_pad > 1:
        print(f"chunk_pad={chunk_pad} (pad FLA chunk boundaries during nocache forward)")

    results: dict[str, dict] = {}
    for name in benchmarks:
        benchmark = instantiate_benchmark(name, limit=limit)
        prompts = [(i, (condition, prompt)) for i, prompt in enumerate(benchmark.prompts)]
        generations = [""] * len(prompts)

        for gen_id, text in probe_generate(
            ckpt,
            iter(prompts),
            max_new_tokens=max_tokens,
            max_seq_len=effective_max,
            chunk_pad=chunk_pad,
        ):
            generations[gen_id] = text

        results[name] = benchmark.compute_metrics(generations)

        if dump_samples > 0:
            out = samples_out or (ckpt_dir / f"samples_{name.lower()}.jsonl")
            _write_sample_records(
                out,
                ckpt_dir=ckpt_dir,
                benchmark_name=name,
                benchmark=benchmark,
                generations=generations,
                dump_samples=dump_samples,
            )
            print(f"Wrote {min(dump_samples, len(generations))} sample generations to {out}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a saved probe checkpoint (GSM8k, MATH, …).")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Directory from --save-ckpt-dir / probe_ckpt_slug")
    parser.add_argument("--benchmarks", default="GSM8k", help="Comma-separated benchmark names (evaluation.benchmarks)")
    parser.add_argument("--limit", type=int, default=50, help="Max problems per benchmark (0 = full set)")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--batch-size", type=int, default=4, help="Unused (nocache generates one prompt at a time)")
    parser.add_argument("--max-context", type=int, default=2048, help="KV cache / prompt buffer size")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate per problem")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--condition",
        default="synth,cot",
        help="HRM condition prefix (matches evaluation/config/hrm_benchmarking.yaml for math)",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write metrics JSON")
    parser.add_argument(
        "--dump-samples",
        type=int,
        default=0,
        help="Write first N prompts/generations/scores to JSONL (see --samples-out)",
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=None,
        help="JSONL path for --dump-samples (default: <ckpt-dir>/samples_<benchmark>.jsonl)",
    )
    parser.add_argument(
        "--chunk-pad",
        type=int,
        default=0,
        help="Pad sequence length to multiple of N for FLA chunk inference (0=off, try 64)",
    )
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    benchmarks = parse_benchmarks(args.benchmarks)

    print(f"ckpt_dir={args.ckpt_dir}")
    print(f"benchmarks={benchmarks}, limit={limit}, batch_size={args.batch_size}, max_context={args.max_context}")

    results = run_benchmarks(
        args.ckpt_dir,
        benchmarks=benchmarks,
        device=args.device,
        batch_size=args.batch_size,
        max_context=args.max_context,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        condition=args.condition,
        limit=limit,
        dump_samples=args.dump_samples,
        samples_out=args.samples_out,
        chunk_pad=args.chunk_pad,
    )

    print("\n" + "#" * 50 + "\nPROBE BENCHMARK SUMMARY\n" + "#" * 50)
    for name, metrics in results.items():
        print(f"\n--- {name} ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<25}: {value:.4f}")
            else:
                print(f"{key:.<25}: {value}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
