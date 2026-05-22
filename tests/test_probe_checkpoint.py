import importlib.util
import json
from pathlib import Path

import torch


def _load_probe_checkpoint():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "probe_checkpoint.py"
    spec = importlib.util.spec_from_file_location("probe_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_full_stack():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 25 - Full Stack Comparison" / "full_stack_comparison.py"
    spec = importlib.util.spec_from_file_location("full_stack_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_tokenizer_info():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "probe_tokenizer_info.py"
    spec = importlib.util.spec_from_file_location("probe_tokenizer_info", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_probe_tokenizer_info_has_hrm_fields():
    tok = _load_tokenizer_info()
    repo = Path(__file__).resolve().parents[1]
    bpe = repo.parent / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json"
    if not bpe.is_file():
        return
    info = tok.default_probe_tokenizer_info(bpe, vocab_size=1000)
    assert "condition_mapping" in info
    assert info["boq"] == tok.DEFAULT_BOQ
    assert "synth" in info["condition_mapping"]
    assert info["vocab_size"] == 1000


def test_probe_checkpoint_roundtrip(tmp_path):
    ckpt_mod = _load_probe_checkpoint()
    fs = _load_full_stack()
    device = torch.device("cpu")
    vocab_size = 512
    token_permutation = torch.arange(vocab_size, dtype=torch.long)

    model, config = fs.build_probe_model(
        variant="dense-attention",
        device=device,
        vocab_size=vocab_size,
        prefix_len=8,
        causal_len=8,
        hidden_size=64,
        vocab_modes=64,
        hidden_modes=32,
        fourier_mode=32,
        pom_order=2,
        token_permutation=token_permutation,
    )

    repo = Path(__file__).resolve().parents[1]
    bpe = repo.parent / "data_io" / "trained_tokenizers" / "bpe" / "tokenizer.json"
    tok_mod = _load_tokenizer_info()
    if not bpe.is_file():
        tokenizer_info = {
            "tokenizer_path": "dummy",
            "boq": tok_mod.DEFAULT_BOQ,
            "eoq": tok_mod.DEFAULT_EOQ,
            "eoa": tok_mod.DEFAULT_EOA,
            "condition_mapping": tok_mod.DEFAULT_CONDITION_MAPPING,
            "vocab_size": vocab_size,
        }
    else:
        tokenizer_info = tok_mod.default_probe_tokenizer_info(bpe, vocab_size=vocab_size)

    out = tmp_path / "dense-attention_h64_s1"
    ckpt_mod.save_probe_checkpoint(
        out,
        model,
        model_config=config,
        probe_meta={"variant": "dense-attention", "seed": 1, "hidden_size": 64},
        tokenizer_info=tokenizer_info,
        token_permutation=token_permutation,
    )

    assert (out / ckpt_mod.STATE_DICT_NAME).is_file()
    meta = json.loads((out / ckpt_mod.PROBE_META_NAME).read_text(encoding="utf-8"))
    assert meta["variant"] == "dense-attention"

    loaded, _cfg, loaded_meta = ckpt_mod.load_probe_checkpoint(out, device=device)
    assert loaded_meta["variant"] == "dense-attention"
    before = sum(p.sum().item() for p in model.parameters())
    after = sum(p.sum().item() for p in loaded.parameters())
    assert before == after


def test_probe_ckpt_slug():
    ckpt_mod = _load_probe_checkpoint()
    assert ckpt_mod.probe_ckpt_slug(variant="a/b", hidden_size=256, seed=3) == "a_b_h256_s3"
