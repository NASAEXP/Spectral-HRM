import importlib.util
from pathlib import Path


def _load_installer():
    path = Path(__file__).resolve().parents[1] / "colab" / "install_fla_colab.py"
    spec = importlib.util.spec_from_file_location("install_fla_colab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pick_triton_version_mapping():
    installer = _load_installer()

    assert installer.pick_triton_version.__defaults__ is None

    original = installer._torch_major_minor
    try:
        installer._torch_major_minor = lambda: (2, 7)
        assert installer.pick_triton_version() == "3.3.1"
        installer._torch_major_minor = lambda: (2, 6)
        assert installer.pick_triton_version() == "3.2.0"
        installer._torch_major_minor = lambda: (2, 5)
        assert installer.pick_triton_version() == "3.2.0"
    finally:
        installer._torch_major_minor = original
