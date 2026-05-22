import torch
import torch.nn.functional as F
from torch import nn

from models.lm_head import LMHead


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


def test_asymmetric_hybrid_embed_differs_from_logit_weight():
    import models.lm_head as lm_head

    vocab = lm_head.AsymmetricHybridFourierLowRankVocab(
        vocab_size=31,
        hidden_size=13,
        init_std=0.25,
        vocab_modes=7,
        hidden_modes=5,
        residual_rank=4,
        residual_scale=0.5,
        bias=True,
    )
    token_ids = torch.tensor([0, 4, 12, 30])
    embed_weight = vocab.embed_weight()
    logit_weight = vocab.logit_weight()

    assert embed_weight.shape == (31, 13)
    assert logit_weight.shape == (31, 13)
    assert not torch.allclose(embed_weight, logit_weight)
    assert torch.allclose(
        vocab.embed(token_ids),
        F.embedding(token_ids, embed_weight) * vocab.embedding_scale,
    )
    hidden = vocab.embed(token_ids)
    assert torch.allclose(vocab.logits(hidden), F.linear(hidden, logit_weight, vocab.bias))


def test_lm_head_asymmetric_hybrid_does_not_share_weight_across_embed_and_logits():
    import models.lm_head as lm_head

    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "hybrid_fourier_lowrank_asymmetric",
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

    assert isinstance(model.vocab_head, lm_head.AsymmetricHybridFourierLowRankVocab)
    assert model.vocab_head.asymmetric_embed_logits is True
    assert model.vocab_head.token_residual.grad is not None
    assert logits.shape == (3, 23)


def test_factorized_fourier_vocab_matches_dense_weight_construction():
    import models.lm_head as lm_head

    vocab = lm_head.FactorizedFourierVocab(
        vocab_size=17,
        hidden_size=9,
        init_std=0.25,
        vocab_modes=6,
        hidden_modes=4,
    )
    token_ids = torch.tensor([0, 2, 9, 16])
    dense = vocab.dense_weight()
    embedded = vocab.embed(token_ids)
    logits = vocab.logits(embedded)

    assert dense.shape == (17, 9)
    assert torch.allclose(embedded, F.embedding(token_ids, dense) * vocab.embedding_scale)
    assert torch.allclose(logits, F.linear(embedded, dense), rtol=1e-5, atol=1e-5)


def test_factorized_fourier_forward_without_materializing_dense_weight():
    import models.lm_head as lm_head

    vocab = lm_head.FactorizedFourierVocab(
        vocab_size=17,
        hidden_size=9,
        init_std=0.25,
        vocab_modes=6,
        hidden_modes=4,
    )
    calls = 0
    original = vocab.dense_weight

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    vocab.dense_weight = counted
    token_ids = torch.tensor([0, 2, 9, 16])
    hidden = vocab.embed(token_ids)
    _logits = vocab.logits(hidden)
    assert calls == 0


def test_multiscale_fourier_vocab_sums_multiple_coefficient_grids():
    import models.lm_head as lm_head

    vocab = lm_head.MultiScaleFourierVocab(
        vocab_size=21,
        hidden_size=10,
        init_std=0.25,
        multiscale_specs=[(5, 3), (7, 4)],
    )
    dense = vocab.dense_weight()
    assert dense.shape == (21, 10)
    assert len(vocab.coefficient_banks) == 2
    assert sum(bank.numel() for bank in vocab.coefficient_banks) < 21 * 10


def test_cluster_hybrid_residual_uses_fewer_token_parameters_than_full_rank():
    import models.lm_head as lm_head

    vocab = lm_head.ClusterHybridFourierLowRankVocab(
        vocab_size=40,
        hidden_size=11,
        init_std=0.25,
        vocab_modes=8,
        hidden_modes=5,
        residual_rank=4,
        num_clusters=8,
        residual_scale=0.5,
    )
    full_rank_params = 40 * 4 + 4 * 11
    cluster_params = 8 * 4 + 4 * 11
    assert vocab.cluster_residual.numel() + vocab.hidden_residual.numel() == cluster_params
    assert cluster_params < full_rank_params
    assert vocab.dense_weight().shape == (40, 11)


def test_tiered_hot_token_vocab_adds_dense_rows_for_hot_tokens():
    import models.lm_head as lm_head

    vocab = lm_head.TieredHotTokenFourierVocab(
        vocab_size=24,
        hidden_size=8,
        init_std=0.25,
        vocab_modes=6,
        hidden_modes=4,
        hot_token_count=5,
        bias=True,
    )
    dense = vocab.dense_weight()
    hot_delta = vocab.hot_token_weight
    ordered = vocab.ordered_dense_weight()
    hot_token_id = int(vocab.token_permutation[0])
    assert hot_delta.shape == (5, 8)
    assert dense.shape == (24, 8)
    assert torch.allclose(dense[hot_token_id], ordered[0] + hot_delta[0])


def test_lm_head_registers_new_vocab_design_types():
    import models.lm_head as lm_head

    configs = [
        ("hybrid_fourier_lowrank_asymmetric", lm_head.AsymmetricHybridFourierLowRankVocab),
        ("factorized_fourier", lm_head.FactorizedFourierVocab),
        ("multiscale_fourier", lm_head.MultiScaleFourierVocab),
        ("cluster_hybrid_fourier_lowrank", lm_head.ClusterHybridFourierLowRankVocab),
        ("tiered_hot_fourier", lm_head.TieredHotTokenFourierVocab),
    ]
    for head_type, expected_cls in configs:
        extra = {}
        if head_type == "hybrid_fourier_lowrank_asymmetric":
            extra = {"residual_rank": 3, "residual_scale": 0.5}
        elif head_type == "multiscale_fourier":
            extra = {"multiscale_specs": [(6, 4), (8, 5)]}
        elif head_type == "cluster_hybrid_fourier_lowrank":
            extra = {"residual_rank": 3, "num_clusters": 4}
        elif head_type == "tiered_hot_fourier":
            extra = {"hot_token_count": 4}
        model = LMHead(
            IdentityModel(hidden_size=12),
            {
                "vocab_size": 23,
                "vocab_head": {
                    "type": head_type,
                    "vocab_modes": 6,
                    "hidden_modes": 5,
                    **extra,
                },
            },
        )
        _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
        logits.sum().backward()
        assert isinstance(model.vocab_head, expected_cls)
