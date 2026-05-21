import importlib.util
from pathlib import Path

import numpy as np


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 4 - Tiny Sapient Format" / "tiny_sapient_probe.py"
    spec = importlib.util.spec_from_file_location("tiny_sapient_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_tiny_dataset_writes_expected_files(tmp_path):
    probe = _load_probe_module()

    dataset_path = probe.build_tiny_dataset(tmp_path / "tiny", num_examples=4, inst_len=5, resp_len=4)

    assert (dataset_path / "metadata.json").exists()
    assert (dataset_path / "tokens.npy").exists()
    assert (dataset_path / "epoch_0" / "inst_start.npy").exists()
    assert np.load(dataset_path / "epoch_0" / "resp_len.npy").tolist() == [4, 4, 4, 4]


def test_tiny_dataset_loads_through_v1dataset(tmp_path):
    probe = _load_probe_module()

    dataset_path = probe.build_tiny_dataset(tmp_path / "tiny", num_examples=4, inst_len=5, resp_len=4)
    batch, scalars, metadata = probe.load_one_v1_batch(dataset_path, batch_max_length=32)

    assert metadata.vocab_size == 512
    assert batch["inputs"].numel() == 32
    assert batch["labels"].numel() == 32
    assert scalars["total_seqlen"] > 0
