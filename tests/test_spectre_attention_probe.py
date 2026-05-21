import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 7 - SPECTRE Attention Mixer" / "spectre_attention_probe.py"
    spec = importlib.util.spec_from_file_location("spectre_attention_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_order():
    probe = _load_probe_module()

    assert probe.parse_variants("dense,spectre") == ["dense", "spectre"]


def test_spectre_variant_sets_token_mixer():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectre",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        spectre_buckets=8,
    )

    assert config["token_mixer"] == "spectre"
    assert config["spectre_num_buckets"] == 8


def test_spectre_fourier_all_variant_sets_both_knobs():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="spectre-fourier-all",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        spectre_buckets=8,
        fourier_mode=32,
    )

    assert config["token_mixer"] == "spectre"
    assert config["fourier_linear"] == {
        "enabled": True,
        "target": "all",
        "in_modes": 32,
        "out_modes": 32,
    }
