import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 24 - FLA GDN Speed Pass" / "fla_gdn_speed_pass.py"
    spec = importlib.util.spec_from_file_location("fla_gdn_speed_pass", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_fla_variant_order():
    probe = _load_probe_module()

    assert probe.parse_variants("pom-sla,pom-gdn,pom-fla-gdn") == [
        "pom-sla",
        "pom-gdn",
        "pom-fla-gdn",
    ]


def test_fla_gdn_config_keeps_pom_l_level():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="pom-fla-gdn",
        vocab_size=260,
        seq_len=96,
        hidden_size=64,
        vocab_modes=128,
        hidden_modes=32,
        fourier_mode=32,
        pom_order=4,
    )

    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "fla_gdn"


def test_fla_gdn_config_disables_h_level_fourier_linear():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="pom-fla-gdn",
        vocab_size=260,
        seq_len=96,
        hidden_size=64,
        vocab_modes=128,
        hidden_modes=32,
        fourier_mode=32,
        pom_order=4,
    )

    assert config["fourier_linear"]["enabled"] is True
    assert config["H_override"]["fourier_linear"]["enabled"] is False


def test_speed_metrics_separate_warmup_from_measured_steps():
    probe = _load_probe_module()

    metrics = probe.compute_speed_metrics(
        train_elapsed_s=2.0,
        warmup_elapsed_s=10.0,
        steps=4,
        warmup_steps=2,
        numseqs=2,
        prefix_len=3,
        causal_len=5,
    )

    assert metrics["warmup_elapsed_s"] == 10.0
    assert metrics["warmup_steps"] == 2
    assert metrics["train_elapsed_s"] == 2.0
    assert metrics["ms_per_step"] == 500.0
    assert metrics["tokens_per_second"] == 32.0
