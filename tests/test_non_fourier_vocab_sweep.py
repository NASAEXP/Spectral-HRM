import importlib.util
from pathlib import Path


def _load_sweep():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 30 - Non-Fourier Vocab Heads" / "non_fourier_vocab_sweep.py"
    spec = importlib.util.spec_from_file_location("non_fourier_vocab_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_non_fourier_config_uses_fla_gdn_and_projected_head():
    sweep = _load_sweep()
    config = sweep.make_config(
        variant="spectral-projected-dense-tied",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        lowrank_rank=32,
        hot_token_count=128,
        vocab_factor_a=32,
        vocab_factor_b=32,
        hidden_factor_a=8,
        hidden_factor_b=16,
    )
    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["vocab_head"]["type"] == "projected_dense_tied"


def test_kronecker_factors_in_config():
    sweep = _load_sweep()
    config = sweep.make_config(
        variant="spectral-tied-kronecker",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        lowrank_rank=32,
        hot_token_count=128,
        vocab_factor_a=32,
        vocab_factor_b=32,
        hidden_factor_a=8,
        hidden_factor_b=16,
    )
    assert config["vocab_head"]["type"] == "tied_kronecker"
    assert config["vocab_head"]["vocab_factor_a"] == 32
    assert config["vocab_head"]["hidden_factor_b"] == 16
