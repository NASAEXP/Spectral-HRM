import importlib.util
from pathlib import Path


def _load_sweep_module():
    repo_root = Path(__file__).resolve().parents[1]
    sweep_path = repo_root / "experiments" / "Experiment 1 - Fourier MLP Weights" / "local_mode_sweep.py"
    spec = importlib.util.spec_from_file_location("local_mode_sweep", sweep_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_modes_accepts_comma_separated_values():
    sweep = _load_sweep_module()

    assert sweep.parse_modes("16, 24,48") == [16, 24, 48]


def test_format_result_row_includes_dense_baseline_and_mode():
    sweep = _load_sweep_module()

    row = sweep.format_result_row(
        "fourier-24",
        {
            "first_eval": 6.0,
            "final_eval": 3.5,
            "train_loss": 3.4,
            "num_params": 12345,
            "peak_vram_mb": 22.25,
            "elapsed_s": 1.25,
        },
    )

    assert row.startswith("fourier-24:")
    assert "eval 6.0000 -> 3.5000" in row
    assert "params=12,345" in row
