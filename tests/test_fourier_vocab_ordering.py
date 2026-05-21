import importlib.util
from pathlib import Path

import torch


def _load_ordering_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 13 - Fourier Vocab Ordering" / "fourier_vocab_ordering.py"
    spec = importlib.util.spec_from_file_location("fourier_vocab_ordering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_13_defaults_include_signal_orderings():
    experiment = _load_ordering_module()
    orderings = experiment.parse_orderings(experiment.DEFAULT_ORDERINGS)

    assert orderings == ["identity", "frequency", "byte_category", "byte_category_frequency", "random"]


def test_frequency_ordering_sorts_common_tokens_first():
    experiment = _load_ordering_module()
    tokens = torch.tensor([3, 3, 3, 1, 1, 4, 2])

    permutation = experiment.make_token_permutation("frequency", tokens=tokens, vocab_size=6)

    assert permutation.tolist()[:6] == [3, 1, 2, 4, 0, 5]


def test_byte_category_ordering_groups_local_byte_tokens():
    experiment = _load_ordering_module()

    permutation = experiment.make_token_permutation("byte_category", tokens=torch.tensor([0]), vocab_size=130)

    positions = {int(token_id): idx for idx, token_id in enumerate(permutation.tolist())}
    assert positions[0] < positions[10]  # reserved before newline byte token
    assert positions[10] < positions[49]  # whitespace before digit byte token '0' + 1
    assert positions[49] < positions[66]  # digits before uppercase 'A' + 1
    assert positions[66] < positions[98]  # uppercase before lowercase 'a' + 1


def test_random_ordering_is_deterministic_and_complete():
    experiment = _load_ordering_module()

    first = experiment.make_token_permutation("random", tokens=torch.tensor([0]), vocab_size=16)
    second = experiment.make_token_permutation("random", tokens=torch.tensor([0]), vocab_size=16)

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(16))
    assert not torch.equal(first, torch.arange(16))
