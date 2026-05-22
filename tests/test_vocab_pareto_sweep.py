import importlib.util
from pathlib import Path


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    sweep_path = repo_root / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "vocab_pareto_sweep.py"
    spec = importlib.util.spec_from_file_location("vocab_pareto_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_multiscale_specs():
    sweep = _load_sweep_module()

    assert sweep.parse_multiscale_specs("512,64;256,128") == [(512, 64), (256, 128)]
    assert sweep.parse_multiscale_specs("") == []


def test_pareto_variant_uses_attention_not_fla_gdn():
    sweep = _load_sweep_module()

    config = sweep.make_config(
        variant="pom-tied-fourier",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        vocab_modes=32,
        hidden_modes=64,
        fourier_mode=32,
        pom_order=4,
        residual_scale=0.5,
        multiscale_specs=[],
        hot_token_count=128,
        num_clusters=64,
    )

    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "attention"
    assert config["H_override"]["fourier_linear"]["enabled"] is False
    assert config["vocab_head"]["type"] == "tied_fourier"
    assert config["vocab_head"]["bias"] is True


def test_asymmetric_hybrid_variant_config():
    sweep = _load_sweep_module()

    config = sweep.make_config(
        variant="pom-asymmetric-hybrid-r16",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        vocab_modes=32,
        hidden_modes=64,
        fourier_mode=32,
        pom_order=4,
        residual_scale=0.5,
        multiscale_specs=[],
        hot_token_count=128,
        num_clusters=64,
    )

    assert config["vocab_head"] == {
        "type": "hybrid_fourier_lowrank_asymmetric",
        "vocab_modes": 32,
        "hidden_modes": 64,
        "residual_rank": 16,
        "residual_scale": 0.5,
        "bias": True,
    }


def test_multiscale_variant_uses_specs():
    sweep = _load_sweep_module()

    config = sweep.make_config(
        variant="pom-multiscale-fourier",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        vocab_modes=32,
        hidden_modes=64,
        fourier_mode=32,
        pom_order=4,
        residual_scale=0.5,
        multiscale_specs=[(32, 16), (48, 24)],
        hot_token_count=128,
        num_clusters=64,
    )

    assert config["vocab_head"]["type"] == "multiscale_fourier"
    assert config["vocab_head"]["multiscale_specs"] == [(32, 16), (48, 24)]
