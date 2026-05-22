import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "scripts" / "load_probe_tokens.py"
    spec = importlib.util.spec_from_file_location("load_probe_tokens", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_laptop_slice_exists_and_loads():
    mod = _load_module()
    slice_dir = mod.GRAM_ROOT / "data_io" / "data_laptop_hrm_slice"
    if not (slice_dir / "metadata.json").is_file():
        return

    tokens, source = mod.load_probe_tokens(min_tokens=10_000, prefer_slice=True)
    assert source.startswith("hrm_slice:")
    assert tokens.numel() >= 10_000
    meta = (slice_dir / "metadata.json").read_text(encoding="utf-8")
    assert "gsm8k" not in meta
    assert "num_examples" in meta


def test_slice_v1_layout_has_epoch_indices():
    slice_dir = Path(__file__).resolve().parents[1].parent / "data_io" / "data_laptop_hrm_slice"
    if not (slice_dir / "epoch_0" / "inst_start.npy").is_file():
        return
    inst_start = np.load(slice_dir / "epoch_0" / "inst_start.npy")
    assert inst_start.shape[0] > 1000
