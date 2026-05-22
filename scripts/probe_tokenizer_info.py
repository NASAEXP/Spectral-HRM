"""HRM-compatible tokenizer_info for probe checkpoints and generation."""

from __future__ import annotations

from pathlib import Path

# Matches data_io/tokenizer defaults (see data_io/tokenizer/src/main.rs).
DEFAULT_CONDITION_MAPPING = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}
DEFAULT_BOQ = "<|im_start|>"
DEFAULT_EOQ = "<|im_end|>"
DEFAULT_EOA = "<|box_end|>"


def resolve_tokenizer_load_path(tokenizer_path: Path) -> str:
    if tokenizer_path.is_dir():
        return str(tokenizer_path)
    if tokenizer_path.name == "tokenizer.json":
        return str(tokenizer_path.parent)
    return str(tokenizer_path)


def default_probe_tokenizer_info(
    tokenizer_path: Path,
    *,
    vocab_size: int | None = None,
) -> dict:
    path = Path(tokenizer_path)
    if vocab_size is None:
        from tokenizers import Tokenizer

        file_path = path / "tokenizer.json" if path.is_dir() else path
        vocab_size = Tokenizer.from_file(str(file_path)).get_vocab_size(True)

    return {
        "tokenizer_path": resolve_tokenizer_load_path(path),
        "boq": DEFAULT_BOQ,
        "eoq": DEFAULT_EOQ,
        "eoa": DEFAULT_EOA,
        "condition_mapping": dict(DEFAULT_CONDITION_MAPPING),
        "vocab_size": vocab_size,
    }
