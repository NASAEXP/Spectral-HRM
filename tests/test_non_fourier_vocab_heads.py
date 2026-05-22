import torch
import torch.nn.functional as F

from models.lm_head import (
    LMHead,
    ProjectedDenseTiedVocab,
    TieredHotLowRankVocab,
    TiedKroneckerVocab,
    TiedLowRankVocab,
    UntiedDenseVocab,
    UntiedLowRankVocab,
)


class IdentityModel(torch.nn.Module):
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


def test_projected_dense_tie_logit_weight_is_embed_times_proj():
    vocab = ProjectedDenseTiedVocab(vocab_size=32, hidden_size=16, init_std=0.25, bias=True)
    expected = vocab.weight @ vocab.proj
    assert torch.allclose(vocab.logit_weight(), expected)


def test_tied_lowrank_matches_explicit_product():
    vocab = TiedLowRankVocab(
        vocab_size=40,
        hidden_size=12,
        init_std=0.25,
        lowrank_rank=5,
        token_order="identity",
    )
    dense = vocab.dense_weight()
    assert dense.shape == (40, 12)
    assert torch.allclose(dense, vocab.token_factors @ vocab.hidden_factors)


def test_tied_kronecker_logits_match_materialized_kron():
    vocab = TiedKroneckerVocab(
        vocab_size=32,
        hidden_size=16,
        init_std=0.25,
        vocab_factor_a=8,
        vocab_factor_b=4,
        hidden_factor_a=4,
        hidden_factor_b=4,
        token_order="identity",
    )
    hidden = torch.randn(3, 16)
    logits = vocab.logits(hidden)
    materialized = torch.kron(vocab.factor_a, vocab.factor_b)
    expected = F.linear(hidden, materialized, vocab.bias)
    assert torch.allclose(logits, expected, atol=1e-4, rtol=1e-4)


def test_untied_dense_uses_different_embed_and_logit_weights():
    vocab = UntiedDenseVocab(vocab_size=24, hidden_size=10, init_std=0.25, bias=True)
    assert not torch.allclose(vocab.input_vocab.weight, vocab.output_vocab.weight)


def test_lm_head_projected_dense_is_asymmetric():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 20,
            "vocab_head": {"type": "projected_dense_tied", "bias": True},
        },
    )
    assert isinstance(model.vocab_head, ProjectedDenseTiedVocab)
    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 3, 5])})
    logits.sum().backward()
    assert model.vocab_head.weight.grad is not None
    assert model.vocab_head.proj.grad is not None
    assert logits.shape == (3, 20)


def test_tiered_hot_lowrank_adds_hot_rows():
    vocab = TieredHotLowRankVocab(
        vocab_size=30,
        hidden_size=11,
        init_std=0.25,
        lowrank_rank=4,
        hot_token_count=6,
        token_order="identity",
    )
    ordered = vocab.ordered_dense_weight()
    cold = vocab.token_factors @ vocab.hidden_factors
    assert torch.allclose(ordered[6:], cold[6:])
    assert not torch.allclose(ordered[:6], cold[:6])
