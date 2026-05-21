import importlib.util
from pathlib import Path


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    sweep_path = repo_root / "experiments" / "Experiment 9 - Vocab Mode Sweep" / "vocab_mode_sweep.py"
    spec = importlib.util.spec_from_file_location("vocab_mode_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_mode_pairs_keeps_order():
    sweep = _load_sweep_module()

    assert sweep.parse_mode_pairs("80x32,160x64") == [(80, 32), (160, 64)]


def test_label_row_includes_vocab_modes():
    sweep = _load_sweep_module()

    assert sweep.label_row("tied-fourier-vocab", 160, 64) == "tied-fourier-vocab@160x64"
