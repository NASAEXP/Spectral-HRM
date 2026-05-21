import importlib.util
from pathlib import Path


def _load_size_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 3 - Local Size Scaling" / "local_size_scaling.py"
    spec = importlib.util.spec_from_file_location("local_size_scaling", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_int_list_accepts_size_values():
    probe = _load_size_module()

    assert probe.parse_int_list("96, 128,256") == [96, 128, 256]


def test_format_summary_line_reports_hidden_size_and_delta():
    probe = _load_size_module()

    line = probe.format_summary_line(
        hidden_size=128,
        dense_eval=3.0,
        fourier_eval=2.8,
        dense_params=200000,
        fourier_params=100000,
    )

    assert line.startswith("hidden=128:")
    assert "delta=-0.2000" in line
    assert "param_ratio=50.0%" in line
