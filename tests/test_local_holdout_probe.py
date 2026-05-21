import importlib.util
from pathlib import Path

import torch


def _load_holdout_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 2 - Local Holdout Probe" / "local_holdout_probe.py"
    spec = importlib.util.spec_from_file_location("local_holdout_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_int_list_accepts_commas_and_spaces():
    probe = _load_holdout_module()

    assert probe.parse_int_list("1, 2,5") == [1, 2, 5]


def test_split_train_eval_tokens_uses_holdout_tail():
    probe = _load_holdout_module()
    tokens = torch.arange(100)

    train_tokens, eval_tokens = probe.split_train_eval_tokens(tokens, eval_fraction=0.2)

    assert train_tokens.tolist() == list(range(80))
    assert eval_tokens.tolist() == list(range(80, 100))


def test_summarize_results_groups_by_variant():
    probe = _load_holdout_module()
    rows = [
        {"variant": "dense", "final_eval": 3.0, "num_params": 100.0},
        {"variant": "dense", "final_eval": 5.0, "num_params": 100.0},
        {"variant": "fourier-64", "final_eval": 4.0, "num_params": 60.0},
    ]

    summary = probe.summarize_results(rows)

    assert summary["dense"]["mean_final_eval"] == 4.0
    assert summary["dense"]["runs"] == 2
    assert summary["fourier-64"]["mean_num_params"] == 60.0
