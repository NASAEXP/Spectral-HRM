import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 22 - FLA GDN Kernel Probe" / "fla_gdn_probe.py"
    spec = importlib.util.spec_from_file_location("fla_gdn_probe", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classifies_missing_fla_first():
    probe = _load_probe_module()

    assert probe.classify_status(fla_available=False, triton_available=False, gdn_import_ok=False) == "missing_fla"


def test_classifies_missing_triton_after_fla():
    probe = _load_probe_module()

    assert probe.classify_status(fla_available=True, triton_available=False, gdn_import_ok=False) == "missing_triton"


def test_classifies_ready_only_when_layer_imports():
    probe = _load_probe_module()

    assert probe.classify_status(fla_available=True, triton_available=True, gdn_import_ok=True) == "ready"
