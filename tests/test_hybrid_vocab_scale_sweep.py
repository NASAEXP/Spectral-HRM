import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 28 - Hybrid Vocab Scale Sweep" / "hybrid_vocab_scale_sweep.py"
    spec = importlib.util.spec_from_file_location("hybrid_vocab_scale_sweep", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_scale_order():
    probe = _load_probe_module()

    assert probe.parse_variants(
        "spectral-hybrid-r128-s025,spectral-hybrid-r128-s050,spectral-hybrid-r128-s100,spectral-hybrid-r128-s200"
    ) == [
        "spectral-hybrid-r128-s025",
        "spectral-hybrid-r128-s050",
        "spectral-hybrid-r128-s100",
        "spectral-hybrid-r128-s200",
    ]


def test_scale_variant_sets_rank128_and_scale():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectral-hybrid-r128-s200",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
    )

    assert config["token_mixer"] == "pom"
    assert config["vocab_head"] == {
        "type": "hybrid_fourier_lowrank",
        "vocab_modes": 512,
        "hidden_modes": 64,
        "residual_rank": 128,
        "residual_scale": 2.0,
        "bias": True,
    }
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["H_override"]["fourier_linear"]["enabled"] is False


def test_dense_tied_anchor_reuses_spectral_body():
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
    )

    assert config["token_mixer"] == "pom"
    assert config["vocab_head"]["type"] == "dense_tied"
    assert config["H_override"]["token_mixer"] == "fla_gdn"
