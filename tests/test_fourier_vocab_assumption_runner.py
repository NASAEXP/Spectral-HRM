import importlib.util
from pathlib import Path


def _load_assumption_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 12 - Fourier Vocab Assumption Tests" / "fourier_vocab_assumption_tests.py"
    spec = importlib.util.spec_from_file_location("fourier_vocab_assumption_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_12_defaults_compare_basis_order_and_checkpoint():
    experiment = _load_assumption_module()
    variants = experiment.parse_variants(experiment.DEFAULT_VARIANTS)

    assert "tied-fourier-vocab-bias" in variants
    assert "tied-fourier-vocab-bias-fft-basis" in variants
    assert "tied-fourier-vocab-reordered" in variants
    assert "tied-fourier-vocab-checkpoint" in variants
    assert "fourier-all-tied-fourier-vocab-bias" in variants
    assert "fourier-all-tied-fourier-vocab-bias-fft-basis" in variants
