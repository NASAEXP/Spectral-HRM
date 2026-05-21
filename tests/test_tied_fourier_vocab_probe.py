import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 8 - Tied Fourier Vocab Matrix" / "tied_fourier_vocab_probe.py"
    spec = importlib.util.spec_from_file_location("tied_fourier_vocab_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_variants_keeps_order():
    probe = _load_probe_module()

    assert probe.parse_variants("dense,tied-fourier-vocab") == ["dense", "tied-fourier-vocab"]
    assert probe.parse_variants("dense-tied-vocab,untied-fourier-vocab") == ["dense-tied-vocab", "untied-fourier-vocab"]


def test_tied_fourier_variant_sets_vocab_head_config():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="tied-fourier-vocab",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )

    assert config["vocab_head"] == {
        "type": "tied_fourier",
        "vocab_modes": 80,
        "hidden_modes": 32,
    }


def test_fft_basis_variants_set_basis_type():
    probe = _load_probe_module()

    tied = probe.make_config(
        variant="tied-fourier-vocab-fft-basis",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    fourier_all = probe.make_config(
        variant="fourier-all-tied-fourier-vocab-fft-basis",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    bias = probe.make_config(
        variant="tied-fourier-vocab-bias-fft-basis",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )

    assert tied["vocab_head"]["basis_type"] == "fft"
    assert fourier_all["fourier_linear"]["target"] == "all"
    assert fourier_all["vocab_head"]["basis_type"] == "fft"
    assert bias["vocab_head"]["basis_type"] == "fft"
    assert bias["vocab_head"]["bias"] is True


def test_fourier_all_tied_vocab_variant_sets_both_compressions():
    probe = _load_probe_module()

    config = probe.make_config(
        variant="fourier-all-tied-fourier-vocab",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )

    assert config["fourier_linear"]["target"] == "all"
    assert config["vocab_head"]["type"] == "tied_fourier"


def test_bias_scale_reorder_and_checkpoint_variants_set_knobs():
    probe = _load_probe_module()

    bias = probe.make_config(
        variant="tied-fourier-vocab-bias",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    scale = probe.make_config(
        variant="tied-fourier-vocab-learned-scale",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    reordered = probe.make_config(
        variant="tied-fourier-vocab-reordered",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    checkpointed = probe.make_config(
        variant="tied-fourier-vocab-checkpoint",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )

    assert bias["vocab_head"]["bias"] is True
    assert scale["vocab_head"]["embedding_scale"] == "learned"
    assert reordered["vocab_head"]["token_order"] == "reverse"
    assert checkpointed["vocab_head"]["checkpoint_weight"] is True


def test_learned_token_and_untied_variants_set_head_types():
    probe = _load_probe_module()

    untied = probe.make_config(
        variant="untied-fourier-vocab",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )
    learned = probe.make_config(
        variant="learned-token-fourier-vocab",
        vocab_size=260,
        seq_len=24,
        hidden_size=64,
        vocab_modes=80,
        hidden_modes=32,
        fourier_mode=32,
    )

    assert untied["vocab_head"]["type"] == "untied_fourier"
    assert learned["vocab_head"]["type"] == "learned_token_fourier"
