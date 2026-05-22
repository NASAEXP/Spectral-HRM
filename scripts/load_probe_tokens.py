"""Load tokens for Spectral-HRM experiment probes (README fallback or real HRM slice)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAM_ROOT = REPO_ROOT.parent
DEFAULT_TOKENIZER_PATH = GRAM_ROOT / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json"
DEFAULT_SLICE_DIR = GRAM_ROOT / "data_io" / "data_laptop_hrm_slice"


def _local_readme_text() -> str:
    parts = []
    for path in (REPO_ROOT / "README.md", REPO_ROOT / "dataset_new.py", REPO_ROOT / "models" / "layers.py"):
        parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(parts)


def load_readme_tokens(tokenizer_path: Path, *, min_tokens: int) -> torch.Tensor:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode(_local_readme_text(), add_special_tokens=False)
    tokens = torch.tensor(encoded.ids, dtype=torch.long)
    if tokens.numel() < min_tokens + 1:
        repeats = (min_tokens + 1 + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    return tokens


def load_slice_tokens(slice_dir: Path, *, min_tokens: int) -> torch.Tensor:
    flat_path = slice_dir / "tokens_flat.npy"
    if not flat_path.is_file():
        flat_path = slice_dir / "tokens.npy"
    if not flat_path.is_file():
        raise FileNotFoundError(f"No tokens_flat.npy or tokens.npy under {slice_dir}")

    tokens = torch.from_numpy(np.load(flat_path, mmap_mode="r").astype(np.int64, copy=False))
    if tokens.numel() < min_tokens + 1:
        repeats = (min_tokens + 1 + tokens.numel() - 1) // tokens.numel()
        tokens = tokens.repeat(repeats)
    return tokens.contiguous()


def load_probe_tokens(
    *,
    min_tokens: int,
    tokenizer_path: Path | None = None,
    slice_dir: Path | None = None,
    prefer_slice: bool = True,
) -> tuple[torch.Tensor, str]:
    tokenizer_path = DEFAULT_TOKENIZER_PATH if tokenizer_path is None else tokenizer_path
    slice_dir = DEFAULT_SLICE_DIR if slice_dir is None else slice_dir

    if prefer_slice and slice_dir.is_dir() and (slice_dir / "metadata.json").is_file():
        return load_slice_tokens(slice_dir, min_tokens=min_tokens), f"hrm_slice:{slice_dir}"

    return load_readme_tokens(tokenizer_path, min_tokens=min_tokens), "readme_fallback"


def split_train_eval_tokens(tokens: torch.Tensor, eval_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1.")
    split_at = int(tokens.numel() * (1.0 - eval_fraction))
    return tokens[:split_at].contiguous(), tokens[split_at:].contiguous()
