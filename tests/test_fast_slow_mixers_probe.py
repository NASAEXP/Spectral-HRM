import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 17 - Fast Slow Mixers" / "fast_slow_mixers.py"
    spec = importlib.util.spec_from_file_location("fast_slow_mixers", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_order():
    probe = _load_probe_module()

    assert probe.parse_variants("attention-attention,pom-attention,pom-spectre") == [
        "attention-attention",
        "pom-attention",
        "pom-spectre",
    ]


def test_fast_slow_config_sets_l_and_h_mixers():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="pom-spectre",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=128,
        hidden_modes=32,
        fourier_mode=32,
        pom_order=4,
        spectre_buckets=8,
    )

    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "spectre"
    assert config["pom_order"] == 4
    assert config["spectre_num_buckets"] == 8
