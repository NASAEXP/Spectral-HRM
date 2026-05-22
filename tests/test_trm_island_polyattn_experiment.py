import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 27 - TRM Island PolyAttn" / "trm_island_polyattn.py"
    spec = importlib.util.spec_from_file_location("trm_island_polyattn", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_island_order():
    probe = _load_probe_module()

    assert probe.parse_variants(
        "spectral-dense-tied-deep,spectral-trm-island-attention,spectral-trm-island-polyattn"
    ) == [
        "spectral-dense-tied-deep",
        "spectral-trm-island-attention",
        "spectral-trm-island-polyattn",
    ]


def test_polyattn_island_config_keeps_pom_and_fla_gdn_outside_island():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectral-trm-island-polyattn",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
        n_layers=8,
        trm_island_every=4,
        trm_island_steps=2,
    )

    assert config["n_layers"] == 8
    assert config["token_mixer"] == "pom"
    assert config["vocab_head"]["type"] == "dense_tied"
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["H_override"]["trm_island_every"] == 4
    assert config["H_override"]["trm_island_mixer"] == "polyattn"
    assert config["H_override"]["trm_island_steps"] == 2
    assert config["H_override"]["fourier_linear"]["enabled"] is False
