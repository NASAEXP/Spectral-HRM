import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 18 - H-Level SLA" / "h_level_sla.py"
    spec = importlib.util.spec_from_file_location("h_level_sla", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_order():
    probe = _load_probe_module()

    assert probe.parse_variants("pom-attention,pom-sla,pom-spectre") == [
        "pom-attention",
        "pom-sla",
        "pom-spectre",
    ]


def test_h_level_sla_config_keeps_pom_l_level():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="pom-sla",
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
    assert config["H_override"]["token_mixer"] == "sla"
