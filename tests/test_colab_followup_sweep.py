import importlib.util
from pathlib import Path


def _load_followup():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "colab_followup_sweep.py"
    spec = importlib.util.spec_from_file_location("colab_followup_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_fla_port():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 29 - Vocab Head Pareto Sweep" / "fla_gdn_port_sweep.py"
    spec = importlib.util.spec_from_file_location("fla_gdn_port_sweep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_hidden_modes_list():
    followup = _load_followup()

    assert followup.parse_hidden_modes_list("64,128,256") == [64, 128, 256]


def test_multiscale_specs_track_hidden_modes():
    followup = _load_followup()

    specs = followup.multiscale_specs_for_hidden_modes(hidden_modes=128, hidden_size=256, vocab_modes=512)
    assert specs == [(512, 128), (256, 256)]


def test_fla_port_config_uses_fla_gdn_on_h_level():
    fla = _load_fla_port()

    config = fla.make_config(
        variant="spectral-multiscale-fourier",
        vocab_size=1024,
        seq_len=64,
        hidden_size=128,
        vocab_modes=32,
        hidden_modes=64,
        fourier_mode=32,
        pom_order=4,
        residual_scale=0.5,
        multiscale_specs=[(32, 16), (24, 32)],
        hot_token_count=64,
        num_clusters=32,
    )

    assert config["token_mixer"] == "pom"
    assert config["H_override"]["token_mixer"] == "fla_gdn"
    assert config["vocab_head"]["type"] == "multiscale_fourier"
