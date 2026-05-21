import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 21 - Gated DeltaNet H-Level" / "gdn_h_level.py"
    spec = importlib.util.spec_from_file_location("gdn_h_level", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_order():
    probe = _load_probe_module()

    assert probe.parse_variants("pom-sla,pom-deltanet,pom-gdn") == [
        "pom-sla",
        "pom-deltanet",
        "pom-gdn",
    ]


def test_gdn_config_keeps_pom_l_level():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="pom-gdn",
        vocab_size=260,
        seq_len=96,
        hidden_size=64,
        vocab_modes=128,
        hidden_modes=32,
        fourier_mode=32,
        pom_order=4,
    )

    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "gdn"
