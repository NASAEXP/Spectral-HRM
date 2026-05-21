import importlib.util
from pathlib import Path


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[1]
    probe_path = repo_root / "experiments" / "Experiment 19 - Long Context H-Level Validation" / "long_context_h_level.py"
    spec = importlib.util.spec_from_file_location("long_context_h_level", probe_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_contexts_accepts_prefix_x_causal_pairs():
    probe = _load_probe_module()

    assert probe.parse_contexts("12x12,24x24,48x48") == [(12, 12), (24, 24), (48, 48)]


def test_summarize_by_context_keeps_variant_rows_separate():
    probe = _load_probe_module()
    rows = [
        {"context": "12x12", "variant": "pom-attention", "final_eval": 7.0, "num_params": 10.0, "peak_vram_mb": 1.0, "elapsed_s": 2.0, "train_elapsed_s": 1.0, "ms_per_step": 10.0, "tokens_per_second": 100.0},
        {"context": "12x12", "variant": "pom-attention", "final_eval": 5.0, "num_params": 10.0, "peak_vram_mb": 3.0, "elapsed_s": 4.0, "train_elapsed_s": 2.0, "ms_per_step": 20.0, "tokens_per_second": 200.0},
        {"context": "24x24", "variant": "pom-sla", "final_eval": 6.0, "num_params": 11.0, "peak_vram_mb": 2.0, "elapsed_s": 3.0, "train_elapsed_s": 1.5, "ms_per_step": 15.0, "tokens_per_second": 150.0},
    ]

    summary = probe.summarize_by_context(rows)

    assert summary[("12x12", "pom-attention")]["runs"] == 2
    assert summary[("12x12", "pom-attention")]["mean_final_eval"] == 6.0
    assert summary[("24x24", "pom-sla")]["mean_tokens_per_second"] == 150.0
