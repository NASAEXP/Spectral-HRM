"""Install Triton builds compatible with Colab PyTorch, then verify FLA GDN import."""

from __future__ import annotations

import re
import subprocess
import sys


FLA_TRITON_INDEX = "https://pypi.fla-org.com/simple"


def _torch_major_minor() -> tuple[int, int]:
    import torch

    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return (2, 6)
    return int(match.group(1)), int(match.group(2))


def pick_triton_version() -> str:
    major, minor = _torch_major_minor()
    if (major, minor) >= (2, 7):
        return "3.3.1"
    if (major, minor) >= (2, 6):
        return "3.2.0"
    return "3.2.0"


def run_pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def install_fla_stack() -> str:
    import torch

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    triton_version = pick_triton_version()
    print(f"installing triton=={triton_version} from FLA index")

    run_pip("uninstall", "-y", "triton", "pytorch-triton", "pytorch-triton-rocm")
    run_pip("install", "-q", f"triton=={triton_version}", "--index-url", FLA_TRITON_INDEX)
    run_pip("install", "-q", "einops", "ninja")
    run_pip("install", "-q", "-r", "requirements-fla.txt")

    return triton_version


def main() -> None:
    install_fla_stack()

    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[1])

    probe = f"{repo_root}/experiments/Experiment 22 - FLA GDN Kernel Probe/fla_gdn_probe.py"
    print("\n--- probe ---", flush=True)
    subprocess.check_call([sys.executable, probe, "--require-ready"])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("\nInstall failed. See COLAB.md troubleshooting.", flush=True)
        raise SystemExit(exc.returncode) from exc
