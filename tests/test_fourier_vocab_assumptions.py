import torch
from torch import nn

from models.lm_head import LMHead, TiedFourierVocab


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


def test_tied_fourier_dense_weight_is_temporary_not_a_parameter():
    vocab = TiedFourierVocab(
        vocab_size=17,
        hidden_size=9,
        init_std=0.25,
        vocab_modes=6,
        hidden_modes=4,
    )

    dense_weight = vocab.dense_weight()
    dense_weight.square().mean().backward()

    assert isinstance(dense_weight, torch.Tensor)
    assert not isinstance(dense_weight, nn.Parameter)
    assert list(dict(vocab.named_parameters())) == ["coefficients"]
    assert vocab.coefficients.grad is not None


def test_tied_fourier_vocab_supports_fft_basis_and_backprop():
    vocab = TiedFourierVocab(
        vocab_size=17,
        hidden_size=9,
        init_std=0.25,
        vocab_modes=7,
        hidden_modes=5,
        basis_type="fft",
    )

    dense_weight = vocab.dense_weight()
    loss = vocab.logits(vocab.embed(torch.tensor([0, 3, 16]), weight=dense_weight), weight=dense_weight).sum()
    loss.backward()

    assert vocab.basis_type == "fft"
    assert dense_weight.shape == (17, 9)
    assert vocab.vocab_basis.shape == (17, 7)
    assert vocab.hidden_basis.shape == (5, 9)
    assert vocab.coefficients.grad is not None


def test_dct_and_fft_basis_choices_are_actually_different():
    common = dict(vocab_size=17, hidden_size=9, init_std=0.25, vocab_modes=7, hidden_modes=5)
    dct_vocab = TiedFourierVocab(**common, basis_type="dct")
    fft_vocab = TiedFourierVocab(**common, basis_type="fft")

    assert not torch.allclose(dct_vocab.vocab_basis, fft_vocab.vocab_basis)
    assert not torch.allclose(dct_vocab.hidden_basis, fft_vocab.hidden_basis)


def test_lm_head_threads_fft_basis_config_into_tied_vocab():
    model = LMHead(
        IdentityModel(hidden_size=12),
        {
            "vocab_size": 23,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 6,
                "hidden_modes": 5,
                "basis_type": "fft",
            },
        },
    )

    _carry, logits = model(carry=None, batch={"inputs": torch.tensor([1, 5, 9])})
    logits.sum().backward()

    assert isinstance(model.vocab_head, TiedFourierVocab)
    assert model.vocab_head.basis_type == "fft"
    assert model.vocab_head.coefficients.grad is not None


def test_checkpoint_weight_recomputes_dense_weight_during_backward():
    vocab = TiedFourierVocab(
        vocab_size=17,
        hidden_size=9,
        init_std=0.25,
        vocab_modes=6,
        hidden_modes=4,
        checkpoint_weight=True,
    )
    vocab.train()

    calls = 0
    original = vocab.ordered_dense_weight

    def counted_ordered_dense_weight(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    vocab.ordered_dense_weight = counted_ordered_dense_weight
    dense_weight = vocab.dense_weight()
    vocab.logits(vocab.embed(torch.tensor([1, 2, 3]), weight=dense_weight), weight=dense_weight).sum().backward()

    assert calls >= 2
    assert vocab.coefficients.grad is not None
