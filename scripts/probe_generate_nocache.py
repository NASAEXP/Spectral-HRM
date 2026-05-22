"""Greedy generation without KV cache (PoM / FLA-GDN / SLA probes cannot use cached decode)."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from simple_inference_engine import InferenceCheckpoint


def _model_max_seq_len(model: nn.Module) -> int:
    core = getattr(model, "model", model)
    h_level = getattr(core, "H_level", None)
    if h_level is not None and hasattr(h_level, "core"):
        return int(h_level.core.config.max_seq_len)
    return 2048


def _pad_for_chunk(token_ids: np.ndarray, *, chunk_size: int, pad_id: int) -> tuple[np.ndarray, int]:
    """Pad tail with pad_id so length is divisible by chunk_size (FLA chunk kernels)."""
    real_len = int(token_ids.shape[0])
    if chunk_size <= 1:
        return token_ids, real_len
    pad_len = (-real_len) % chunk_size
    if pad_len == 0:
        return token_ids, real_len
    padded = np.concatenate([token_ids, np.full(pad_len, pad_id, dtype=token_ids.dtype)])
    return padded, real_len


def _forward_logits(
    model: nn.Module,
    token_ids: np.ndarray,
    *,
    prefix_len: int,
    device: torch.device,
    chunk_pad: int = 0,
    pad_id: int = 0,
) -> torch.Tensor:
    real_len = int(token_ids.shape[0])
    forward_ids, predict_index = _pad_for_chunk(token_ids, chunk_size=chunk_pad, pad_id=pad_id)
    total_len = int(forward_ids.shape[0])
    causal_len = total_len - prefix_len
    inputs = torch.from_numpy(forward_ids).to(device=device, dtype=torch.long)
    batch = {
        "inputs": inputs,
        "position_ids": torch.arange(total_len, device=device, dtype=torch.long),
        "prefix_lens": torch.tensor([prefix_len], dtype=torch.int32, device=device),
        "causal_lens": torch.tensor([causal_len], dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor([0, total_len], dtype=torch.int32, device=device),
        "total_seqlen": torch.tensor(total_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(1, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(prefix_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(causal_len, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(total_len, dtype=torch.int64, device=device),
    }
    _carry, logits = model(carry=None, batch=batch, bp_steps=2)
    return logits[predict_index - 1]


@torch.inference_mode()
def probe_generate_nocache(
    ckpt: InferenceCheckpoint,
    iterator: Iterator[tuple[int, tuple[str, str]]],
    *,
    max_new_tokens: int,
    max_seq_len: int | None = None,
    chunk_pad: int = 0,
) -> Iterator[tuple[int, str]]:
    model = ckpt.model
    device = next(model.parameters()).device
    hard_max = min(max_seq_len or _model_max_seq_len(model), max_new_tokens + 4096)
    stop_token: int = ckpt.tokenizer.convert_tokens_to_ids(ckpt.tokenizer_info["eoa"])  # pyright: ignore[reportAssignmentType]
    pad_id: int = ckpt.tokenizer.convert_tokens_to_ids("<|PAD|>")  # pyright: ignore[reportAssignmentType]
    if pad_id is None:
        pad_id = 0

    pending = list(iterator)
    for prompt_id, (condition, prompt) in tqdm(pending, desc="probe_generate"):
        prompt_tokens = ckpt.tokenize_prompt(condition, prompt)
        if prompt_tokens.size >= hard_max:
            yield prompt_id, ""
            continue

        prefix_len = int(prompt_tokens.size)
        sequence = np.array(prompt_tokens, dtype=np.int64)

        for _ in range(max_new_tokens):
            if sequence.shape[0] >= hard_max:
                break
            logits = _forward_logits(
                model,
                sequence,
                prefix_len=prefix_len,
                device=device,
                chunk_pad=chunk_pad,
                pad_id=pad_id,
            )
            next_id = int(logits.argmax(dim=-1).item())
            sequence = np.append(sequence, next_id)
            if next_id == stop_token:
                break

        gen_tokens = sequence[prefix_len:]
        yield prompt_id, ckpt.decode_generation(gen_tokens, stop_token)
