import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 25 - Full Stack Comparison" / "full_stack_comparison.py"
    spec = importlib.util.spec_from_file_location("full_stack_comparison", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_full_stack_order():
    probe = _load_probe_module()

    assert probe.parse_variants(
        "dense-attention,dense-tied-attention,fourier-pom-sla-tied-fourier,fourier-pom-fla-gdn-tied-fourier,fourier-pom-fla-gdn-dense-tied"
    ) == [
        "dense-attention",
        "dense-tied-attention",
        "fourier-pom-sla-tied-fourier",
        "fourier-pom-fla-gdn-tied-fourier",
        "fourier-pom-fla-gdn-dense-tied",
    ]


def test_dense_attention_config_is_originalish_control():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="dense-attention",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
    )

    assert "fourier_linear" not in config
    assert "vocab_head" not in config
    assert config["token_mixer"] == "attention"
    assert config["H_override"]["token_mixer"] == "attention"


def test_dense_tied_attention_config_controls_vocab_tying_only():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="dense-tied-attention",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
    )

    assert "fourier_linear" not in config
    assert config["vocab_head"]["type"] == "dense_tied"
    assert config["token_mixer"] == "attention"
    assert config["H_override"]["token_mixer"] == "attention"


def test_fla_tied_fourier_config_keeps_compressed_vocab():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="fourier-pom-fla-gdn-tied-fourier",
        vocab_size=65536,
        seq_len=256,
        hidden_size=256,
        vocab_modes=512,
        hidden_modes=64,
        fourier_mode=64,
        pom_order=4,
    )

    assert config["token_mixer"] == "pom"
    assert config["vocab_head"]["type"] == "tied_fourier"
    assert config["vocab_head"]["bias"] is True
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["H_override"]["fourier_linear"]["enabled"] is False


def test_fla_dense_tied_config_uses_dense_tied_vocab():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="fourier-pom-fla-gdn-dense-tied",
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
    assert config["H_override"]["fourier_linear"]["enabled"] is False
