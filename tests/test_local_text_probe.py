import importlib.util
from pathlib import Path

import torch

from models.common import IGNORE_LABEL_ID


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 1 - Fourier MLP Weights" / "local_text_probe.py"
    spec = importlib.util.spec_from_file_location("local_text_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_byte_tokenize_keeps_ids_inside_vocab():
    probe = _load_probe_module()

    tokens = probe.byte_tokenize("Hello", vocab_size=260)

    assert tokens.dtype == torch.long
    assert tokens.tolist() == [73, 102, 109, 109, 112]
    assert int(tokens.min()) >= 1
    assert int(tokens.max()) < 260


def test_make_prefixlm_batch_masks_prefix_and_predicts_response():
    probe = _load_probe_module()
    tokens = torch.arange(1, 65, dtype=torch.long)

    batch = probe.make_prefixlm_batch(
        tokens,
        offset=0,
        numseqs=2,
        prefix_len=4,
        causal_len=4,
        device=torch.device("cpu"),
    )

    assert batch["inputs"].shape == (16,)
    assert batch["labels"].shape == (16,)
    assert batch["cu_seqlens"].tolist() == [0, 8, 16]
    assert batch["labels"][:3].eq(IGNORE_LABEL_ID).all()
    assert batch["labels"][3].item() == 5
    assert batch["labels"][7].item() == 9
