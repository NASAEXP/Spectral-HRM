import importlib.util
from pathlib import Path

import torch


def _load_ordering_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 14 - Tokenizer-Aware Vocab Ordering" / "tokenizer_aware_vocab_ordering.py"
    spec = importlib.util.spec_from_file_location("tokenizer_aware_vocab_ordering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_14_defaults_include_tokenizer_aware_orderings():
    experiment = _load_ordering_module()
    orderings = experiment.parse_orderings(experiment.DEFAULT_ORDERINGS)

    assert orderings == ["bpe_rank", "token_frequency", "token_category", "token_category_frequency", "random"]


def test_experiment_14_locks_token_frequency_as_carry_forward_ordering():
    experiment = _load_ordering_module()

    assert experiment.LOCKED_ORDERING == "token_frequency"
    assert experiment.LOCKED_CONTROL == "bpe_rank"
    assert experiment.parse_orderings(experiment.LOCKED_ORDERINGS) == ["bpe_rank", "token_frequency"]


def test_token_text_category_is_tokenizer_aware():
    experiment = _load_ordering_module()

    assert experiment.token_text_category("<|PAD|>") < experiment.token_text_category("Ġthe")
    assert experiment.token_text_category("Ġthe") == experiment.token_text_category("hello")
    assert experiment.token_text_category("123") < experiment.token_text_category("hello")
    assert experiment.token_text_category("!") > experiment.token_text_category("hello")


def test_token_category_order_groups_by_token_text():
    experiment = _load_ordering_module()
    id_to_token = ["<|PAD|>", "!", "Ġthe", "123", "abc", "é"]

    permutation = experiment.make_token_permutation(
        "token_category",
        tokens=torch.tensor([4, 4, 2, 3, 1]),
        id_to_token=id_to_token,
    )

    assert permutation.tolist() == [0, 3, 2, 4, 1, 5]


def test_token_category_frequency_keeps_categories_then_frequency():
    experiment = _load_ordering_module()
    id_to_token = ["<|PAD|>", "Ġz", "Ġa", "9", "1", "."]
    tokens = torch.tensor([2, 2, 2, 1, 4, 4, 3, 5])

    permutation = experiment.make_token_permutation(
        "token_category_frequency",
        tokens=tokens,
        id_to_token=id_to_token,
    )

    assert permutation.tolist() == [0, 4, 3, 2, 1, 5]


def test_bpe_rank_order_is_identity():
    experiment = _load_ordering_module()
    id_to_token = ["<|PAD|>", "!", "Ġthe", "123"]

    permutation = experiment.make_token_permutation(
        "bpe_rank",
        tokens=torch.tensor([2, 2, 3]),
        id_to_token=id_to_token,
    )

    assert torch.equal(permutation, torch.arange(len(id_to_token)))
