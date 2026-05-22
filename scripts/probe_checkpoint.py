"""Save and load Spectral-HRM experiment probe weights for downstream benchmarks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import AutoTokenizer

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead, TiedFourierVocab


def _load_probe_tokenizer_info():
    path = Path(__file__).resolve().parent / "probe_tokenizer_info.py"
    spec = importlib.util.spec_from_file_location("probe_tokenizer_info", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def default_probe_tokenizer_info(*args, **kwargs):
    return _load_probe_tokenizer_info().default_probe_tokenizer_info(*args, **kwargs)

if TYPE_CHECKING:
    from simple_inference_engine import InferenceCheckpoint

PROBE_CKPT_VERSION = 1
MODEL_CONFIG_NAME = "model_config.json"
PROBE_META_NAME = "probe_meta.json"
TOKENIZER_INFO_NAME = "tokenizer_info.json"
STATE_DICT_NAME = "model.pt"
TOKEN_PERMUTATION_NAME = "token_permutation.pt"


def probe_ckpt_slug(*, variant: str, hidden_size: int, seed: int) -> str:
    safe_variant = variant.replace("/", "_")
    return f"{safe_variant}_h{hidden_size}_s{seed}"


def save_probe_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    *,
    model_config: dict[str, Any],
    probe_meta: dict[str, Any],
    tokenizer_info: dict[str, Any],
    token_permutation: torch.Tensor,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(probe_meta)
    meta["probe_ckpt_version"] = PROBE_CKPT_VERSION

    torch.save(model.state_dict(), ckpt_dir / STATE_DICT_NAME)
    torch.save(token_permutation.cpu(), ckpt_dir / TOKEN_PERMUTATION_NAME)
    (ckpt_dir / MODEL_CONFIG_NAME).write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (ckpt_dir / PROBE_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ckpt_dir / TOKENIZER_INFO_NAME).write_text(json.dumps(tokenizer_info, indent=2), encoding="utf-8")
    return ckpt_dir


def _apply_token_permutation(model: nn.Module, token_permutation: torch.Tensor) -> None:
    for module in model.modules():
        if isinstance(module, TiedFourierVocab):
            module.set_token_permutation(token_permutation)


def build_probe_model(
    model_config: dict[str, Any],
    *,
    device: torch.device,
    token_permutation: torch.Tensor | None = None,
) -> LMHead:
    model = LMHead(HierarchicalReasoningModel(model_config), model_config).to(device)
    if token_permutation is not None:
        _apply_token_permutation(model, token_permutation)
    return model


def load_probe_checkpoint(ckpt_dir: Path, *, device: torch.device) -> tuple[LMHead, dict[str, Any], dict[str, Any]]:
    ckpt_dir = Path(ckpt_dir)
    model_config = json.loads((ckpt_dir / MODEL_CONFIG_NAME).read_text(encoding="utf-8"))
    probe_meta = json.loads((ckpt_dir / PROBE_META_NAME).read_text(encoding="utf-8"))
    state_dict = torch.load(ckpt_dir / STATE_DICT_NAME, map_location=device, weights_only=True)
    token_permutation = torch.load(ckpt_dir / TOKEN_PERMUTATION_NAME, map_location=device, weights_only=True)

    model = build_probe_model(model_config, device=device, token_permutation=token_permutation)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, model_config, probe_meta


def load_probe_inference_checkpoint(ckpt_dir: Path, *, device: torch.device | None = None) -> "InferenceCheckpoint":
    from simple_inference_engine import InferenceCheckpoint

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(ckpt_dir)
    tokenizer_info = json.loads((ckpt_dir / TOKENIZER_INFO_NAME).read_text(encoding="utf-8"))
    model, _model_config, _probe_meta = load_probe_checkpoint(ckpt_dir, device=device)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(dtype=dtype).eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_info["tokenizer_path"], use_fast=True)
    return InferenceCheckpoint(
        model=model,
        carry=None,
        tokenizer=tokenizer,
        tokenizer_info=tokenizer_info,
    )
