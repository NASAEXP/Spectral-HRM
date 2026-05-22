import json
from pathlib import Path


def _notebook_text() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "colab" / "free_fla_gdn_probe.ipynb"
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join(
        line
        for cell in data["cells"]
        for line in cell.get("source", [])
    )


def test_free_colab_notebook_clones_repo_and_tokenizer_source():
    text = _notebook_text()

    assert "git clone --depth 1 https://github.com/NASAEXP/Spectral-HRM.git" in text
    assert "git clone --depth 1 https://github.com/sapientinc/data_io.git" in text


def test_free_colab_notebook_runs_fla_gate_and_tiny_gdn_smoke():
    text = _notebook_text()

    assert "pip install -q -r requirements-fla.txt" in text
    assert "Experiment 22 - FLA GDN Kernel Probe" in text
    assert "Experiment 21 - Gated DeltaNet H-Level" in text
    assert "Experiment 24 - FLA GDN Speed Pass" in text
    assert "Experiment 25 - Full Stack Comparison" in text
    assert "--steps 5 --seeds 1" in text
