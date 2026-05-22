import torch
import torch.nn.functional as F
from torch import nn

from models.lm_head import LMHead


def _tied_fourier_vocab_class():
    import models.lm_head as lm_head

    assert hasattr(lm_head, "TiedFourierVocab"), "TiedFourierVocab is not implemented yet."
    return lm_head.TiedFourierVocab


class IdentityModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head_hint = {
            "in": {"dim": hidden_size, "init_std": 0.25},
            "out": {"dim": hidden_size, "init_std": 0.25},
        }
        self.create_cache = lambda **_kwargs: None
        self.compute_train_extra_args = lambda _state: {}

    def forward(self, carry, x, **_seq_info):
        return carry, x


def test_tied_fourier_vocab_uses_one_matrix_for_embedding_and_logits():
    TiedFourierVocab = _tied_fourier_vocab_class()
    vocab = TiedFourierVocab(
        vocab_size=19,
        hidden_size=11,
        init_std=0.25,
        vocab_modes=5,
        hidden_modes=4,
    )
    token_ids = torch.tensor([0, 3, 7, 18])
    dense_weight = vocab.dense_weight()
    embedded = vocab.embed(token_ids, weight=dense_weight)
    logits = vocab.logits(embedded, weight=dense_weight)

    assert list(dict(vocab.named_parameters())) == ["coefficients"]
    assert dense_weight.shape == (19, 11)
    assert torch.allclose(embedded, F.embedding(token_ids, dense_weight) * vocab.embedding_scale)
    assert torch.allclose(logits, F.linear(embedded, dense_weight))
    assert vocab.coefficients.numel() < 2 * 19 * 11


def test_lm_head_can_use_tied_fourier_vocab():
    TiedFourierVocab = _tied_fourier_vocab_class()
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
            },
        },
    )
    batch = {"inputs": torch.tensor([1, 5, 9])}

    _carry, logits = model(carry=None, batch=batch)
    logits.sum().backward()

    assert isinstance(model.vocab_head, TiedFourierVocab)
    assert model.embed_tokens is model.vocab_head
    assert model.lm_head is model.vocab_head
    assert logits.shape == (3, 23)
    assert model.vocab_head.coefficients.grad is not None


def test_lm_head_generates_tied_fourier_weight_once_per_forward():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
            },
        },
    )
    assert model.vocab_head is not None

    calls = 0
    original_dense_weight = model.vocab_head.dense_weight

    def counted_dense_weight(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_dense_weight(*args, **kwargs)

    model.vocab_head.dense_weight = counted_dense_weight

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})

    assert logits.shape == (3, 23)
    assert calls == 1


def test_lm_head_can_use_dense_tied_vocab():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {"type": "dense_tied"},
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert model.embed_tokens is model.vocab_head
    assert model.lm_head is model.vocab_head
    assert logits.shape == (3, 23)
    assert list(dict(model.vocab_head.named_parameters())) == ["weight"]
    assert model.vocab_head.weight.grad is not None


def test_lm_head_can_use_hybrid_fourier_lowrank_vocab():
    import models.lm_head as lm_head

    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "hybrid_fourier_lowrank",
                "vocab_modes": 6,
                "hidden_modes": 5,
                "residual_rank": 3,
                "residual_scale": 0.5,
                "bias": True,
            },
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert isinstance(model.vocab_head, lm_head.HybridFourierLowRankVocab)
    assert model.embed_tokens is model.vocab_head
    assert model.lm_head is model.vocab_head
    assert logits.shape == (3, 23)
    assert model.vocab_head.dense_weight().shape == (23, 12)
    assert model.vocab_head.coefficients.grad is not None
    assert model.vocab_head.token_residual.grad is not None
    assert model.vocab_head.hidden_residual.grad is not None
    assert model.vocab_head.bias.grad is not None
    assert sum(param.numel() for param in model.vocab_head.parameters()) < 23 * 12 + 23


def test_lm_head_can_use_untied_fourier_vocab():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "untied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
            },
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert logits.shape == (3, 23)
    assert model.vocab_head.input_vocab.coefficients.grad is not None
    assert model.vocab_head.output_vocab.coefficients.grad is not None
    assert model.vocab_head.input_vocab.coefficients is not model.vocab_head.output_vocab.coefficients


def test_vocab_bias_adds_trainable_logit_bias():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
                "bias": True,
            },
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert logits.shape == (3, 23)
    assert model.vocab_head.bias is not None
    assert model.vocab_head.bias.grad is not None


def test_embedding_scale_modes_are_configurable():
    none_scale = _tied_fourier_vocab_class()(
        vocab_size=19,
        hidden_size=16,
        init_std=0.25,
        vocab_modes=5,
        hidden_modes=4,
        embedding_scale="none",
    )
    sqrt_scale = _tied_fourier_vocab_class()(
        vocab_size=19,
        hidden_size=16,
        init_std=0.25,
        vocab_modes=5,
        hidden_modes=4,
        embedding_scale="sqrt_hidden",
    )
    learned_scale = _tied_fourier_vocab_class()(
        vocab_size=19,
        hidden_size=16,
        init_std=0.25,
        vocab_modes=5,
        hidden_modes=4,
        embedding_scale="learned",
    )

    assert none_scale.embedding_scale == 1.0
    assert sqrt_scale.embedding_scale == 4.0
    assert learned_scale.learned_embedding_scale is not None


def test_learned_token_fourier_vocab_uses_learned_vocab_axis():
    import models.lm_head as lm_head

    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "learned_token_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
            },
        },
    )
    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert isinstance(model.vocab_head, lm_head.LearnedTokenFourierVocab)
    assert model.vocab_head.token_basis.grad is not None
    assert model.vocab_head.coefficients.grad is not None


def test_token_reordering_scatter_maps_ordered_rows_back_to_token_ids():
    TiedFourierVocab = _tied_fourier_vocab_class()
    vocab = TiedFourierVocab(
        vocab_size=8,
        hidden_size=6,
        init_std=0.25,
        vocab_modes=4,
        hidden_modes=3,
        token_order="reverse",
    )
    ordered = vocab.ordered_dense_weight()
    dense = vocab.dense_weight()

    assert int(vocab.token_permutation[0]) == 7
    assert torch.allclose(dense[7], ordered[0])
    assert torch.allclose(dense[0], ordered[-1])


def test_tied_fourier_vocab_accepts_custom_token_permutation():
    TiedFourierVocab = _tied_fourier_vocab_class()
    vocab = TiedFourierVocab(
        vocab_size=8,
        hidden_size=6,
        init_std=0.25,
        vocab_modes=4,
        hidden_modes=3,
    )
    custom_permutation = torch.tensor([2, 4, 0, 6, 1, 7, 3, 5])

    vocab.set_token_permutation(custom_permutation)
    ordered = vocab.ordered_dense_weight()
    dense = vocab.dense_weight()

    assert torch.equal(vocab.token_permutation.cpu(), custom_permutation)
    assert torch.allclose(dense[2], ordered[0])
    assert torch.allclose(dense[5], ordered[-1])


def test_checkpointed_tied_fourier_vocab_backward_works():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
                "checkpoint_weight": True,
            },
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert model.vocab_head.checkpoint_weight is True
    assert model.vocab_head.coefficients.grad is not None
