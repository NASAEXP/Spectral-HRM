import importlib.util
from pathlib import Path


def _load_target_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 5 - Fourier Attention Targets" / "fourier_target_sweep.py"
    spec = importlib.util.spec_from_file_location("fourier_target_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_targets_keeps_order():
    sweep = _load_target_module()

    assert sweep.parse_targets("dense,mlp,attention,all") == ["dense", "mlp", "attention", "all"]


def test_variant_name_for_target():
    sweep = _load_target_module()

    assert sweep.variant_name("dense", 64) == "dense"
    assert sweep.variant_name("attention", 64) == "fourier-attention-64"
