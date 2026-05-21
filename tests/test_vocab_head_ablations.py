import importlib.util
from pathlib import Path


def _load_ablation_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 11 - Vocab Head Ablations" / "vocab_head_ablation.py"
    spec = importlib.util.spec_from_file_location("vocab_head_ablation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_variants_include_all_seven_knobs():
    ablation = _load_ablation_module()
    variants = ablation.parse_variants(ablation.DEFAULT_VARIANTS)

    assert "dense-tied-vocab" in variants
    assert "untied-fourier-vocab" in variants
    assert "tied-fourier-vocab-bias" in variants
    assert "tied-fourier-vocab-learned-scale" in variants
    assert "learned-token-fourier-vocab" in variants
    assert "tied-fourier-vocab-reordered" in variants
    assert "tied-fourier-vocab-checkpoint" in variants
