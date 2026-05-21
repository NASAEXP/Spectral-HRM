import importlib.util
from pathlib import Path


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 6 - Target Mode Sweep" / "target_mode_sweep.py"
    spec = importlib.util.spec_from_file_location("target_mode_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_target_modes_accepts_groups():
    sweep = _load_sweep_module()

    assert sweep.parse_target_modes("mlp:48,64;attention:64;all:96,128") == [
        ("mlp", 48),
        ("mlp", 64),
        ("attention", 64),
        ("all", 96),
        ("all", 128),
    ]


def test_select_top_configs_sorts_by_final_eval():
    sweep = _load_sweep_module()
    rows = [
        {"target": "mlp", "mode": 64, "final_eval": 3.1},
        {"target": "attention", "mode": 96, "final_eval": 2.9},
        {"target": "all", "mode": 128, "final_eval": 3.0},
    ]

    assert sweep.select_top_configs(rows, top_k=2) == [("attention", 96), ("all", 128)]
