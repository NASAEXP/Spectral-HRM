import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 26 - Hybrid Fourier Vocab Bridge" / "vocab_bridge_sweep.py"
    spec = importlib.util.spec_from_file_location("vocab_bridge_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_bridge_order():
    probe = _load_probe_module()

    assert probe.parse_variants(
        "spectral-tied-fourier,spectral-hybrid-r16,spectral-hybrid-r64,spectral-hybrid-r128,spectral-dense-tied"
    ) == [
        "spectral-tied-fourier",
        "spectral-hybrid-r16",
        "spectral-hybrid-r64",
        "spectral-hybrid-r128",
        "spectral-dense-tied",
    ]


def test_hybrid_rank_variant_sets_lowrank_bridge_config():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectral-hybrid-r64",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
        residual_scale=0.5,
    )

    assert config["fourier_linear"]["target"] == "all"
    assert config["token_mixer"] == "pom"
    assert config["vocab_head"] == {
        "type": "hybrid_fourier_lowrank",
        "vocab_modes": 512,
        "hidden_modes": 64,
        "residual_rank": 64,
        "residual_scale": 0.5,
        "bias": True,
    }
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["H_override"]["fourier_linear"]["enabled"] is False


def test_dense_tied_anchor_keeps_best_spectral_body():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectral-dense-tied",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
        residual_scale=0.5,
    )

    assert config["token_mixer"] == "pom"
    assert config["vocab_head"]["type"] == "dense_tied"
    assert config["H_override"]["token_mixer"] == "fla_gdn"
