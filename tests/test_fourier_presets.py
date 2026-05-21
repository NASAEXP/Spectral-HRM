from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_arch_config(name: str) -> dict:
    path = REPO_ROOT / "config" / "arch" / "net" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_quality_preset_locks_attention_96():
    config = _load_arch_config("hrm_fourier_attention_quality.yaml")

    assert config["fourier_linear"] == {
        "enabled": True,
        "target": "attention",
        "in_modes": 96,
        "out_modes": 96,
    }


def test_lean_preset_locks_all_128():
    config = _load_arch_config("hrm_fourier_all_lean.yaml")

    assert config["fourier_linear"] == {
        "enabled": True,
        "target": "all",
        "in_modes": 128,
        "out_modes": 128,
    }


def test_spectre_preset_locks_token_mixer():
    config = _load_arch_config("hrm_spectre.yaml")

    assert config["token_mixer"] == "spectre"
    assert config["spectre_num_buckets"] == 16


def test_tied_fourier_vocab_preset_locks_vocab_head():
    config = _load_arch_config("hrm_tied_fourier_vocab.yaml")

    assert config["vocab_head"] == {
        "type": "tied_fourier",
        "vocab_modes": 160,
        "hidden_modes": 64,
    }


def test_fourier_all_tied_vocab_preset_locks_both_compressions():
    config = _load_arch_config("hrm_fourier_all_tied_vocab.yaml")

    assert config["fourier_linear"]["target"] == "all"
    assert config["vocab_head"] == {
        "type": "tied_fourier",
        "vocab_modes": 512,
        "hidden_modes": 64,
        "bias": True,
    }


def test_fourier_all_tied_vocab_checkpoint_preset_locks_4gb_survival_knob():
    config = _load_arch_config("hrm_fourier_all_tied_vocab_checkpoint.yaml")

    assert config["fourier_linear"]["target"] == "all"
    assert config["vocab_head"] == {
        "type": "tied_fourier",
        "vocab_modes": 512,
        "hidden_modes": 64,
        "bias": True,
        "checkpoint_weight": True,
    }


def test_dense_tied_vocab_preset_locks_dense_tying():
    config = _load_arch_config("hrm_dense_tied_vocab.yaml")

    assert config["vocab_head"] == {"type": "dense_tied"}


def test_fourier_all_dense_tied_vocab_preset_locks_body_compression_and_dense_tying():
    config = _load_arch_config("hrm_fourier_all_dense_tied_vocab.yaml")

    assert config["fourier_linear"]["target"] == "all"
    assert config["vocab_head"] == {"type": "dense_tied"}
