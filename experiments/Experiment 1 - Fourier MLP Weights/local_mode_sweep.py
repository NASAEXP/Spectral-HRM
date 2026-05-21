from pathlib import Path
import argparse
import importlib.util
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_text_probe():
    probe_path = Path(__file__).with_name("local_text_probe.py")
    spec = importlib.util.spec_from_file_location("local_text_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_modes(value: str) -> list[int]:
    modes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("At least one mode value is required.")
    return modes


def format_result_row(name: str, result: dict[str, float]) -> str:
    return (
        f"{name}: "
        f"eval {result['first_eval']:.4f} -> {result['final_eval']:.4f}, "
        f"last_train_loss={result['train_loss']:.4f}, "
        f"params={int(result['num_params']):,}, "
        f"peak_vram_mb={result['peak_vram_mb']:.1f}, "
        f"elapsed_s={result['elapsed_s']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Fourier mode counts on the local byte-text probe.")
    parser.add_argument("--modes", default="16,24,32,48,64")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--numseqs", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=24)
    parser.add_argument("--causal-len", type=int, default=24)
    args = parser.parse_args()

    probe = _load_text_probe()
    modes = parse_modes(args.modes)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokens_needed = (args.prefix_len + args.causal_len) * args.numseqs * (args.steps + 8)
    tokens = probe.load_local_text_tokens(vocab_size=260, min_tokens=tokens_needed)

    print(f"device={device}")
    print(f"tokens={tokens.numel():,}, steps={args.steps}, hidden_size={args.hidden_size}, modes={modes}")

    dense = probe.train_variant(
        "dense",
        fourier=False,
        tokens=tokens,
        steps=args.steps,
        device=device,
        hidden_size=args.hidden_size,
        modes=modes[0],
        numseqs=args.numseqs,
        prefix_len=args.prefix_len,
        causal_len=args.causal_len,
    )
    print(format_result_row("dense", dense))

    for mode in modes:
        result = probe.train_variant(
            f"fourier-{mode}",
            fourier=True,
            tokens=tokens,
            steps=args.steps,
            device=device,
            hidden_size=args.hidden_size,
            modes=mode,
            numseqs=args.numseqs,
            prefix_len=args.prefix_len,
            causal_len=args.causal_len,
        )
        print(format_result_row(f"fourier-{mode}", result))


if __name__ == "__main__":
    main()
