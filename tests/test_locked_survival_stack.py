import importlib.util
from pathlib import Path


def _load_experiment_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 16 - Locked Survival Stack" / "locked_survival_stack.py"
    spec = importlib.util.spec_from_file_location("locked_survival_stack", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_16_defaults_lock_2x2_stack():
    experiment = _load_experiment_module()
    variants = experiment.parse_variants(experiment.DEFAULT_VARIANTS)

    assert variants == [
        "control-bpe-rank",
        "frequency-order",
        "checkpoint-control",
        "survival-locked",
    ]


def test_experiment_16_variant_specs_encode_order_and_checkpoint():
    experiment = _load_experiment_module()

    assert experiment.VARIANT_SPECS["control-bpe-rank"] == {
        "ordering": "bpe_rank",
        "base_variant": "fourier-all-tied-fourier-vocab-bias",
    }
    assert experiment.VARIANT_SPECS["survival-locked"] == {
        "ordering": "token_frequency",
        "base_variant": "fourier-all-tied-fourier-vocab-bias-checkpoint",
    }
