"""Kaggle GPU script: Exp 29 hidden_modes sweep + optional FLA port.

Upload this folder as a Kaggle Script kernel, or push via CLI (see kaggle/KAGGLE.md).
Edit kernel-metadata.json: set id to your-kaggle-username/spectral-hrm-exp29
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/Spectral-HRM")
TOKENIZER = Path("/kaggle/working/data_io/trained_tokenizers/bpe/tokenizer.json")

# After Phase A, set best hidden_modes from summary (default 64).
HIDDEN_MODES = 64
RUN_FLA_PORT = True


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def clone_deps() -> None:
    run(["git", "clone", "--depth", "1", "https://github.com/NASAEXP/Spectral-HRM.git", str(REPO)])
    run(["git", "clone", "--depth", "1", "https://github.com/sapientinc/data_io.git", "/kaggle/working/data_io"])


def install_stack() -> bool:
    run([sys.executable, "-m", "pip", "install", "-q", "einops", "pydantic", "tokenizers", "transformers"])
    try:
        run([sys.executable, str(REPO / "colab" / "install_fla_colab.py")], cwd=REPO)
        return True
    except subprocess.CalledProcessError:
        print("FLA/Triton not ready — skipping Phase B (FLA-GDN port).", flush=True)
        return False


def phase_a() -> None:
    run(
        [
            sys.executable,
            str(REPO / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "colab_followup_sweep.py"),
            "--steps",
            "40",
            "--warmup-steps",
            "1",
            "--seeds",
            "1,2,3",
            "--device",
            "cuda",
            "--tokenizer-path",
            str(TOKENIZER),
            "--variants",
            "pom-tied-fourier,pom-multiscale-fourier,pom-tiered-hot-fourier",
            "--hidden-modes-list",
            "64,128,192,256",
        ],
        cwd=REPO,
    )


def phase_b() -> None:
    multiscale = f"512,{HIDDEN_MODES};256,{min(256, HIDDEN_MODES * 2)}"
    run(
        [
            sys.executable,
            str(REPO / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "fla_gdn_port_sweep.py"),
            "--steps",
            "40",
            "--warmup-steps",
            "1",
            "--seeds",
            "1,2,3",
            "--device",
            "cuda",
            "--tokenizer-path",
            str(TOKENIZER),
            "--hidden-modes",
            str(HIDDEN_MODES),
            "--multiscale-specs",
            multiscale,
        ],
        cwd=REPO,
    )


def main() -> None:
    clone_deps()
    fla_ready = install_stack()
    phase_a()
    if RUN_FLA_PORT and fla_ready:
        phase_b()
    print("Done. Download /kaggle/working output from the kernel session.", flush=True)


if __name__ == "__main__":
    main()
