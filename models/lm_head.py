from typing import Literal, Tuple
import math

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pydantic import BaseModel, Field

from models.layers import LinearInit, ScaledEmbeddingInit, Carry, _cosine_basis
from models.common import IGNORE_LABEL_ID, packing_sequence_sum, trunc_normal_init_


class VocabHeadConfig(BaseModel):
    type: Literal[
        "dense",
        "dense_tied",
        "projected_dense_tied",
        "tied_lowrank",
        "tiered_hot_lowrank",
        "tied_kronecker",
        "untied_dense",
        "untied_lowrank",
        "tied_fourier",
        "hybrid_fourier_lowrank",
        "hybrid_fourier_lowrank_asymmetric",
        "factorized_fourier",
        "multiscale_fourier",
        "cluster_hybrid_fourier_lowrank",
        "tiered_hot_fourier",
        "untied_fourier",
        "learned_token_fourier",
    ] = "dense"
    vocab_modes: int = 2048
    hidden_modes: int = 512
    residual_rank: int = 64
    residual_scale: float = 0.5
    basis_type: Literal["dct", "fft"] = "dct"
    bias: bool = False
    embedding_scale: Literal["init", "none", "sqrt_hidden", "learned"] = "init"
    token_order: Literal["identity", "reverse", "even_odd"] = "identity"
    checkpoint_weight: bool = False
    multiscale_specs: list[tuple[int, int]] = Field(default_factory=list)
    num_clusters: int = 256
    hot_token_count: int = 4096
    lowrank_rank: int = 128
    vocab_factor_a: int = 256
    vocab_factor_b: int = 256
    hidden_factor_a: int = 16
    hidden_factor_b: int = 16


class LMHeadConfig(BaseModel):
    vocab_size: int
    vocab_head: VocabHeadConfig = Field(default_factory=VocabHeadConfig)


def _make_token_permutation(vocab_size: int, token_order: str, *, device=None) -> Tensor:
    if token_order == "identity":
        return torch.arange(vocab_size, device=device, dtype=torch.long)
    if token_order == "reverse":
        return torch.arange(vocab_size - 1, -1, -1, device=device, dtype=torch.long)
    if token_order == "even_odd":
        evens = torch.arange(0, vocab_size, 2, device=device, dtype=torch.long)
        odds = torch.arange(1, vocab_size, 2, device=device, dtype=torch.long)
        return torch.cat((evens, odds))
    raise ValueError(f"Unknown token_order: {token_order}")


def _apply_token_permutation(ordered_weight: Tensor, token_permutation: Tensor) -> Tensor:
    if token_permutation.numel() == 0:
        return ordered_weight

    dense_weight = torch.empty_like(ordered_weight)
    dense_weight[token_permutation.to(device=ordered_weight.device)] = ordered_weight
    return dense_weight


def _inverse_token_permutation(token_permutation: Tensor) -> Tensor:
    inverse = torch.empty_like(token_permutation)
    inverse[token_permutation] = torch.arange(token_permutation.numel(), device=token_permutation.device, dtype=torch.long)
    return inverse


def _scatter_ordered_logits(ordered_logits: Tensor, token_permutation: Tensor) -> Tensor:
    inverse = _inverse_token_permutation(token_permutation.to(device=ordered_logits.device))
    return ordered_logits[:, inverse]


def _real_fourier_basis(length: int, modes: int, *, device=None, dtype=None) -> Tensor:
    modes = min(modes, length)
    compute_dtype = torch.float32
    positions = torch.arange(length, device=device, dtype=compute_dtype) / float(length)
    basis = torch.empty((modes, length), device=device, dtype=compute_dtype)
    basis[0].fill_(1.0 / math.sqrt(length))

    row = 1
    frequency = 1
    scale = math.sqrt(2.0 / length)
    while row < modes:
        angle = 2.0 * math.pi * frequency * positions
        basis[row] = torch.cos(angle) * scale
        row += 1
        if row < modes:
            basis[row] = torch.sin(angle) * scale
            row += 1
        frequency += 1

    return basis.to(dtype=dtype) if dtype is not None else basis


def _frequency_basis(length: int, modes: int, basis_type: str, *, device=None, dtype=None) -> Tensor:
    if basis_type == "dct":
        return _cosine_basis(length, modes, device=device, dtype=dtype)
    if basis_type == "fft":
        return _real_fourier_basis(length, modes, device=device, dtype=dtype)
    raise ValueError(f"Unknown basis_type: {basis_type}")


class EmbeddingScaleMixin:
    embedding_scale_mode: str
    embedding_scale_value: float
    learned_embedding_scale: nn.Parameter | None

    def _init_embedding_scale(self, hidden_size: int, init_std: float, mode: str, **kwargs) -> None:
        self.embedding_scale_mode = mode
        if mode == "init":
            value = 1.0 / init_std
        elif mode == "none":
            value = 1.0
        elif mode == "sqrt_hidden":
            value = math.sqrt(hidden_size)
        elif mode == "learned":
            value = 1.0 / init_std
        else:
            raise ValueError(f"Unknown embedding_scale mode: {mode}")

        self.embedding_scale_value = float(value)
        self.learned_embedding_scale = None
        if mode == "learned":
            param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
            self.learned_embedding_scale = nn.Parameter(torch.tensor(float(value), **param_kwargs))

    @property
    def embedding_scale(self) -> Tensor | float:
        return self.learned_embedding_scale if self.learned_embedding_scale is not None else self.embedding_scale_value


class DenseTiedVocab(nn.Module, EmbeddingScaleMixin):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_size, hidden_size), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.weight.to(dtype=dtype) if dtype is not None else self.weight

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


def _require_product(n: int, factors: tuple[int, ...], *, label: str) -> None:
    product = 1
    for factor in factors:
        product *= factor
    if product != n:
        raise ValueError(f"{label} must factor as {'*'.join(map(str, factors))}={product}, got {n}.")


class ProjectedDenseTiedVocab(nn.Module, EmbeddingScaleMixin):
    """Dense tied embed; logits use W_in @ T (decoupled read/write gradients through T)."""

    asymmetric_embed_logits = True

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_size, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.proj = nn.Parameter(
            trunc_normal_init_(torch.empty((hidden_size, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def embed_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.weight.to(dtype=dtype) if dtype is not None else self.weight

    def logit_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.weight.dtype
        return self.weight.to(dtype=dtype) @ self.proj.to(dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.embed_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.logit_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class TiedLowRankVocab(nn.Module, EmbeddingScaleMixin):
    """W ≈ A @ B with A (vocab, rank) and B (rank, hidden); tied embed and logits."""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 lowrank_rank: int,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 **kwargs):
        if lowrank_rank <= 0:
            raise ValueError("lowrank_rank must be positive.")
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lowrank_rank = min(lowrank_rank, vocab_size, hidden_size)
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}
        self.token_factors = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_size, self.lowrank_rank), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.hidden_factors = nn.Parameter(
            trunc_normal_init_(torch.empty((self.lowrank_rank, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.token_permutation = nn.Buffer(_make_token_permutation(vocab_size, token_order, **basis_kwargs), persistent=False)
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def ordered_dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.token_factors.dtype
        return self.token_factors.to(dtype=dtype) @ self.hidden_factors.to(dtype=dtype)

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return _apply_token_permutation(self.ordered_dense_weight(dtype=dtype), self.token_permutation)

    def set_token_permutation(self, token_permutation: Tensor) -> None:
        token_permutation = token_permutation.detach().to(device=self.token_permutation.device, dtype=torch.long)
        if token_permutation.shape != (self.vocab_size,):
            raise ValueError(f"token_permutation must have shape ({self.vocab_size},), got {tuple(token_permutation.shape)}")
        expected = torch.arange(self.vocab_size, device=token_permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(token_permutation).values, expected):
            raise ValueError("token_permutation must contain each token id exactly once.")
        self.token_permutation = token_permutation

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class TieredHotLowRankVocab(TiedLowRankVocab):
    """Low-rank cold bulk plus dense rows for hot tokens in ordered token space."""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 lowrank_rank: int,
                 hot_token_count: int,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            init_std=init_std,
            lowrank_rank=lowrank_rank,
            bias=bias,
            embedding_scale=embedding_scale,
            token_order=token_order,
            **kwargs,
        )
        self.hot_token_count = min(hot_token_count, vocab_size)
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        self.hot_token_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((self.hot_token_count, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def ordered_dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        ordered_weight = super().ordered_dense_weight(dtype=dtype)
        if self.hot_token_count > 0:
            ordered_weight = ordered_weight.clone()
            ordered_weight[: self.hot_token_count] = ordered_weight[: self.hot_token_count] + self.hot_token_weight.to(
                dtype=ordered_weight.dtype
            )
        return ordered_weight


class TiedKroneckerVocab(nn.Module, EmbeddingScaleMixin):
    """W = kron(A, B) with A (v1, h1) and B (v2, h2); vocab=v1*v2, hidden=h1*h2."""

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_factor_a: int,
                 vocab_factor_b: int,
                 hidden_factor_a: int,
                 hidden_factor_b: int,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 **kwargs):
        super().__init__()
        _require_product(vocab_size, (vocab_factor_a, vocab_factor_b), label="vocab_size")
        _require_product(hidden_size, (hidden_factor_a, hidden_factor_b), label="hidden_size")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_factor_a = vocab_factor_a
        self.vocab_factor_b = vocab_factor_b
        self.hidden_factor_a = hidden_factor_a
        self.hidden_factor_b = hidden_factor_b
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}
        self.factor_a = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_factor_a, hidden_factor_a), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.factor_b = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_factor_b, hidden_factor_b), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.token_permutation = nn.Buffer(_make_token_permutation(vocab_size, token_order, **basis_kwargs), persistent=False)
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def set_token_permutation(self, token_permutation: Tensor) -> None:
        token_permutation = token_permutation.detach().to(device=self.token_permutation.device, dtype=torch.long)
        if token_permutation.shape != (self.vocab_size,):
            raise ValueError(f"token_permutation must have shape ({self.vocab_size},), got {tuple(token_permutation.shape)}")
        expected = torch.arange(self.vocab_size, device=token_permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(token_permutation).values, expected):
            raise ValueError("token_permutation must contain each token id exactly once.")
        self.token_permutation = token_permutation

    def _kron_logits_ordered(self, hidden_states: Tensor) -> Tensor:
        hidden_grid = hidden_states.reshape(-1, self.hidden_factor_a, self.hidden_factor_b)
        scores = torch.einsum("bxy,ix,jy->bij", hidden_grid, self.factor_a, self.factor_b)
        return scores.reshape(hidden_states.shape[0], self.vocab_factor_a * self.vocab_factor_b)

    def _kron_embed_rows(self, token_ids: Tensor) -> Tensor:
        token_a = torch.div(token_ids, self.vocab_factor_b, rounding_mode="floor")
        token_b = token_ids % self.vocab_factor_b
        rows_a = self.factor_a[token_a]
        rows_b = self.factor_b[token_b]
        return (rows_a.unsqueeze(2) * rows_b.unsqueeze(1)).reshape(token_ids.shape[0], self.hidden_size)

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.factor_a.dtype
        ordered = torch.kron(self.factor_a.to(dtype=dtype), self.factor_b.to(dtype=dtype))
        return _apply_token_permutation(ordered, self.token_permutation)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is not None:
            return F.embedding(input_ids, weight) * self.embedding_scale
        inverse = _inverse_token_permutation(self.token_permutation)
        ordered_ids = inverse[input_ids.to(device=inverse.device)]
        return self._kron_embed_rows(ordered_ids) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is not None:
            bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
            return F.linear(hidden_states, weight.to(dtype=hidden_states.dtype), bias)
        ordered_logits = self._kron_logits_ordered(hidden_states)
        logits = _scatter_ordered_logits(ordered_logits, self.token_permutation)
        if self.bias is not None:
            logits = logits + self.bias.to(dtype=hidden_states.dtype)
        return logits

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class UntiedDenseVocab(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 **kwargs):
        super().__init__()
        self.input_vocab = DenseTiedVocab(
            vocab_size,
            hidden_size,
            init_std,
            bias=False,
            embedding_scale=embedding_scale,
            **kwargs,
        )
        self.output_vocab = DenseTiedVocab(
            vocab_size,
            hidden_size,
            init_std,
            bias=bias,
            embedding_scale="none",
            **kwargs,
        )

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.input_vocab.dense_weight(dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.input_vocab.embed(input_ids, weight=weight)

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.output_vocab.logits(hidden_states, weight=weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class UntiedLowRankVocab(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 lowrank_rank: int,
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 **kwargs):
        super().__init__()
        self.input_vocab = TiedLowRankVocab(
            vocab_size,
            hidden_size,
            init_std,
            lowrank_rank=lowrank_rank,
            bias=False,
            embedding_scale=embedding_scale,
            token_order=token_order,
            **kwargs,
        )
        self.output_vocab = TiedLowRankVocab(
            vocab_size,
            hidden_size,
            init_std,
            lowrank_rank=lowrank_rank,
            bias=bias,
            embedding_scale="none",
            token_order=token_order,
            **kwargs,
        )

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.input_vocab.dense_weight(dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.input_vocab.embed(input_ids, weight=weight)

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.output_vocab.logits(hidden_states, weight=weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class TiedFourierVocab(nn.Module, EmbeddingScaleMixin):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_modes = min(vocab_modes, vocab_size)
        self.hidden_modes = min(hidden_modes, hidden_size)
        self.basis_type = basis_type
        self.checkpoint_weight = checkpoint_weight

        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}

        coeff_std = init_std * math.sqrt((vocab_size * hidden_size) / (self.vocab_modes * self.hidden_modes))
        self.coefficients = nn.Parameter(
            trunc_normal_init_(torch.empty((self.vocab_modes, self.hidden_modes), **param_kwargs), std=coeff_std)  # pyright: ignore[reportArgumentType]
        )
        self.vocab_basis = nn.Buffer(_frequency_basis(vocab_size, self.vocab_modes, basis_type, **basis_kwargs).T, persistent=False)
        self.hidden_basis = nn.Buffer(_frequency_basis(hidden_size, self.hidden_modes, basis_type, **basis_kwargs), persistent=False)
        self.token_permutation = nn.Buffer(_make_token_permutation(vocab_size, token_order, **basis_kwargs), persistent=False)
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def set_token_permutation(self, token_permutation: Tensor) -> None:
        token_permutation = token_permutation.detach().to(device=self.token_permutation.device, dtype=torch.long)
        if token_permutation.shape != (self.vocab_size,):
            raise ValueError(f"token_permutation must have shape ({self.vocab_size},), got {tuple(token_permutation.shape)}")

        expected = torch.arange(self.vocab_size, device=token_permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(token_permutation).values, expected):
            raise ValueError("token_permutation must contain each token id exactly once.")

        self.token_permutation = token_permutation

    def ordered_dense_weight(self, coefficients: Tensor | None = None, dtype: torch.dtype | None = None) -> Tensor:
        coefficients = self.coefficients if coefficients is None else coefficients
        dtype = dtype or coefficients.dtype
        vocab_basis = self.vocab_basis.to(device=coefficients.device, dtype=dtype)
        hidden_basis = self.hidden_basis.to(device=coefficients.device, dtype=dtype)
        return vocab_basis @ coefficients.to(dtype=dtype) @ hidden_basis

    def _dense_weight_from_coefficients(self, coefficients: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        ordered_weight = self.ordered_dense_weight(coefficients, dtype=dtype)
        return _apply_token_permutation(ordered_weight, self.token_permutation)

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        if self.checkpoint_weight and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda coefficients: self._dense_weight_from_coefficients(coefficients, dtype=dtype),
                self.coefficients,
                use_reentrant=False,
            )
        return self._dense_weight_from_coefficients(self.coefficients, dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is None and self.checkpoint_weight and self.training and torch.is_grad_enabled():
            def _checkpointed_logits(coefficients, hidden_states, bias):
                ordered_weight = self.vocab_basis.to(device=coefficients.device, dtype=hidden_states.dtype) @ coefficients.to(dtype=hidden_states.dtype) @ self.hidden_basis.to(device=coefficients.device, dtype=hidden_states.dtype)
                w = _apply_token_permutation(ordered_weight, self.token_permutation)
                b = bias.to(dtype=hidden_states.dtype) if bias is not None else None
                return F.linear(hidden_states, w, b)
            
            return checkpoint(
                _checkpointed_logits,
                self.coefficients,
                hidden_states,
                self.bias,
                use_reentrant=False,
            )
        
        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class HybridFourierLowRankVocab(TiedFourierVocab):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 residual_rank: int,
                 residual_scale: float = 0.5,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 **kwargs):
        if residual_rank <= 0:
            raise ValueError("residual_rank must be positive.")
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            init_std=init_std,
            vocab_modes=vocab_modes,
            hidden_modes=hidden_modes,
            basis_type=basis_type,
            bias=bias,
            embedding_scale=embedding_scale,
            token_order=token_order,
            checkpoint_weight=checkpoint_weight,
            **kwargs,
        )
        self.residual_rank = min(residual_rank, vocab_size, hidden_size)
        self.residual_scale = float(residual_scale)
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        self.token_residual = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_size, self.residual_rank), **param_kwargs), std=1.0 / math.sqrt(self.residual_rank))  # pyright: ignore[reportArgumentType]
        )
        self.hidden_residual = nn.Parameter(
            trunc_normal_init_(torch.empty((self.residual_rank, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def _lowrank_residual(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.token_residual.dtype
        return (self.token_residual.to(dtype=dtype) @ self.hidden_residual.to(dtype=dtype)) * self.residual_scale

    def _dense_weight_from_parts(self,
                                 coefficients: Tensor,
                                 token_residual: Tensor,
                                 hidden_residual: Tensor,
                                 dtype: torch.dtype | None = None) -> Tensor:
        ordered_weight = self.ordered_dense_weight(coefficients, dtype=dtype)
        fourier_weight = _apply_token_permutation(ordered_weight, self.token_permutation)
        dtype = dtype or coefficients.dtype
        residual = (token_residual.to(dtype=dtype) @ hidden_residual.to(dtype=dtype)) * self.residual_scale
        return fourier_weight + residual

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        if self.checkpoint_weight and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda coefficients, token_residual, hidden_residual: self._dense_weight_from_parts(
                    coefficients,
                    token_residual,
                    hidden_residual,
                    dtype=dtype,
                ),
                self.coefficients,
                self.token_residual,
                self.hidden_residual,
                use_reentrant=False,
            )
        return self._dense_weight_from_parts(self.coefficients, self.token_residual, self.hidden_residual, dtype=dtype)

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is None and self.checkpoint_weight and self.training and torch.is_grad_enabled():
            def _checkpointed_logits(coefficients, token_residual, hidden_residual, hidden_states, bias):
                w = self._dense_weight_from_parts(
                    coefficients,
                    token_residual,
                    hidden_residual,
                    dtype=hidden_states.dtype,
                )
                b = bias.to(dtype=hidden_states.dtype) if bias is not None else None
                return F.linear(hidden_states, w, b)

            return checkpoint(
                _checkpointed_logits,
                self.coefficients,
                self.token_residual,
                self.hidden_residual,
                hidden_states,
                self.bias,
                use_reentrant=False,
            )

        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)


class AsymmetricHybridFourierLowRankVocab(HybridFourierLowRankVocab):
    """Fourier weight for embeddings; Fourier + low-rank residual for logits."""

    asymmetric_embed_logits = True

    def embed_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return _apply_token_permutation(self.ordered_dense_weight(dtype=dtype), self.token_permutation)

    def logit_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.dense_weight(dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.embed_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is not None:
            return F.linear(
                hidden_states,
                weight.to(dtype=hidden_states.dtype),
                self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None,
            )
        if self.checkpoint_weight and self.training and torch.is_grad_enabled():
            def _checkpointed_logits(coefficients, token_residual, hidden_residual, hidden_states, bias):
                w = self._dense_weight_from_parts(
                    coefficients,
                    token_residual,
                    hidden_residual,
                    dtype=hidden_states.dtype,
                )
                b = bias.to(dtype=hidden_states.dtype) if bias is not None else None
                return F.linear(hidden_states, w, b)

            return checkpoint(
                _checkpointed_logits,
                self.coefficients,
                self.token_residual,
                self.hidden_residual,
                hidden_states,
                self.bias,
                use_reentrant=False,
            )

        dense_weight = self.logit_weight(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)


class FactorizedFourierVocab(TiedFourierVocab):
    """Apply Fourier vocab via matmuls without materializing the dense weight matrix."""

    def _ordered_embed_rows(self, ordered_ids: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.coefficients.dtype
        vocab_rows = self.vocab_basis.to(device=ordered_ids.device, dtype=dtype)[ordered_ids]
        hidden_basis = self.hidden_basis.to(device=ordered_ids.device, dtype=dtype)
        coefficients = self.coefficients.to(dtype=dtype)
        return vocab_rows @ coefficients @ hidden_basis

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is not None:
            return F.embedding(input_ids, weight) * self.embedding_scale
        inverse = _inverse_token_permutation(self.token_permutation)
        ordered_ids = inverse[input_ids.to(device=inverse.device)]
        return self._ordered_embed_rows(ordered_ids, dtype=None) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is not None:
            bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
            return F.linear(hidden_states, weight.to(dtype=hidden_states.dtype), bias)
        dtype = hidden_states.dtype
        hidden_basis = self.hidden_basis.to(device=hidden_states.device, dtype=dtype)
        coefficients = self.coefficients.to(dtype=dtype)
        vocab_basis = self.vocab_basis.to(device=hidden_states.device, dtype=dtype)
        hidden = hidden_states @ hidden_basis.T
        hidden = hidden @ coefficients.T
        ordered_logits = hidden @ vocab_basis.T
        logits = _scatter_ordered_logits(ordered_logits, self.token_permutation)
        if self.bias is not None:
            logits = logits + self.bias.to(dtype=dtype)
        return logits


class MultiScaleFourierVocab(nn.Module, EmbeddingScaleMixin):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 multiscale_specs: list[tuple[int, int]],
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 vocab_modes: int = 0,
                 hidden_modes: int = 0,
                 **kwargs):
        if not multiscale_specs:
            multiscale_specs = [(vocab_modes or 2048, hidden_modes or 512)]
        super().__init__()
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.basis_type = basis_type
        self.checkpoint_weight = checkpoint_weight
        self.multiscale_specs = [(min(vm, vocab_size), min(hm, hidden_size)) for vm, hm in multiscale_specs]

        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}
        self.coefficient_banks = nn.ParameterList()
        self._vocab_bases: list[Tensor] = []
        self._hidden_bases: list[Tensor] = []
        for scale_idx, (vocab_modes, hidden_modes) in enumerate(self.multiscale_specs):
            coeff_std = init_std * math.sqrt((vocab_size * hidden_size) / (vocab_modes * hidden_modes))
            self.coefficient_banks.append(
                nn.Parameter(
                    trunc_normal_init_(torch.empty((vocab_modes, hidden_modes), **param_kwargs), std=coeff_std)  # pyright: ignore[reportArgumentType]
                )
            )
            self.register_buffer(
                f"vocab_basis_{scale_idx}",
                _frequency_basis(vocab_size, vocab_modes, basis_type, **basis_kwargs).T,
                persistent=False,
            )
            self.register_buffer(
                f"hidden_basis_{scale_idx}",
                _frequency_basis(hidden_size, hidden_modes, basis_type, **basis_kwargs),
                persistent=False,
            )
            self._vocab_bases.append(getattr(self, f"vocab_basis_{scale_idx}"))
            self._hidden_bases.append(getattr(self, f"hidden_basis_{scale_idx}"))
        self.token_permutation = nn.Buffer(_make_token_permutation(vocab_size, token_order, **basis_kwargs), persistent=False)
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None

    def set_token_permutation(self, token_permutation: Tensor) -> None:
        token_permutation = token_permutation.detach().to(device=self.token_permutation.device, dtype=torch.long)
        if token_permutation.shape != (self.vocab_size,):
            raise ValueError(f"token_permutation must have shape ({self.vocab_size},), got {tuple(token_permutation.shape)}")
        expected = torch.arange(self.vocab_size, device=token_permutation.device, dtype=torch.long)
        if not torch.equal(torch.sort(token_permutation).values, expected):
            raise ValueError("token_permutation must contain each token id exactly once.")
        self.token_permutation = token_permutation

    def ordered_dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.coefficient_banks[0].dtype
        ordered_weight = None
        for coefficients, vocab_basis, hidden_basis in zip(self.coefficient_banks, self._vocab_bases, self._hidden_bases, strict=True):
            vocab_basis = vocab_basis.to(device=coefficients.device, dtype=dtype)
            hidden_basis = hidden_basis.to(device=coefficients.device, dtype=dtype)
            term = vocab_basis @ coefficients.to(dtype=dtype) @ hidden_basis
            ordered_weight = term if ordered_weight is None else ordered_weight + term
        assert ordered_weight is not None
        return ordered_weight

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return _apply_token_permutation(self.ordered_dense_weight(dtype=dtype), self.token_permutation)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class ClusterHybridFourierLowRankVocab(HybridFourierLowRankVocab):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 residual_rank: int,
                 num_clusters: int,
                 residual_scale: float = 0.5,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 **kwargs):
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive.")
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            init_std=init_std,
            vocab_modes=vocab_modes,
            hidden_modes=hidden_modes,
            residual_rank=residual_rank,
            residual_scale=residual_scale,
            basis_type=basis_type,
            bias=bias,
            embedding_scale=embedding_scale,
            token_order=token_order,
            checkpoint_weight=checkpoint_weight,
            **kwargs,
        )
        self.num_clusters = min(num_clusters, vocab_size)
        del self.token_residual
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        self.cluster_residual = nn.Parameter(
            trunc_normal_init_(torch.empty((self.num_clusters, self.residual_rank), **param_kwargs), std=1.0 / math.sqrt(self.residual_rank))  # pyright: ignore[reportArgumentType]
        )
        cluster_ids = torch.arange(vocab_size, dtype=torch.long) * self.num_clusters // vocab_size
        cluster_ids.clamp_(max=self.num_clusters - 1)
        self.register_buffer("cluster_ids", cluster_ids, persistent=False)

    def _lowrank_residual(self, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or self.cluster_residual.dtype
        token_factor = self.cluster_residual.to(dtype=dtype)[self.cluster_ids.to(device=self.cluster_residual.device)]
        return token_factor @ self.hidden_residual.to(dtype=dtype) * self.residual_scale

    def _dense_weight_from_parts(self,
                                 coefficients: Tensor,
                                 cluster_residual: Tensor,
                                 hidden_residual: Tensor,
                                 dtype: torch.dtype | None = None) -> Tensor:
        ordered_weight = self.ordered_dense_weight(coefficients, dtype=dtype)
        fourier_weight = _apply_token_permutation(ordered_weight, self.token_permutation)
        dtype = dtype or coefficients.dtype
        token_factor = cluster_residual.to(dtype=dtype)[self.cluster_ids.to(device=cluster_residual.device)]
        residual = (token_factor @ hidden_residual.to(dtype=dtype)) * self.residual_scale
        return fourier_weight + residual

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        if self.checkpoint_weight and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda coefficients, cluster_residual, hidden_residual: self._dense_weight_from_parts(
                    coefficients,
                    cluster_residual,
                    hidden_residual,
                    dtype=dtype,
                ),
                self.coefficients,
                self.cluster_residual,
                self.hidden_residual,
                use_reentrant=False,
            )
        return self._dense_weight_from_parts(self.coefficients, self.cluster_residual, self.hidden_residual, dtype=dtype)

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is None and self.checkpoint_weight and self.training and torch.is_grad_enabled():
            def _checkpointed_logits(coefficients, cluster_residual, hidden_residual, hidden_states, bias):
                w = self._dense_weight_from_parts(
                    coefficients,
                    cluster_residual,
                    hidden_residual,
                    dtype=hidden_states.dtype,
                )
                b = bias.to(dtype=hidden_states.dtype) if bias is not None else None
                return F.linear(hidden_states, w, b)

            return checkpoint(
                _checkpointed_logits,
                self.coefficients,
                self.cluster_residual,
                self.hidden_residual,
                hidden_states,
                self.bias,
                use_reentrant=False,
            )

        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)


class TieredHotTokenFourierVocab(TiedFourierVocab):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 hot_token_count: int,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 **kwargs):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            init_std=init_std,
            vocab_modes=vocab_modes,
            hidden_modes=hidden_modes,
            basis_type=basis_type,
            bias=bias,
            embedding_scale=embedding_scale,
            token_order=token_order,
            checkpoint_weight=checkpoint_weight,
            **kwargs,
        )
        self.hot_token_count = min(hot_token_count, vocab_size)
        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        self.hot_token_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((self.hot_token_count, hidden_size), **param_kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def _dense_weight_from_coefficients(self, coefficients: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        ordered_weight = self.ordered_dense_weight(coefficients, dtype=dtype)
        if self.hot_token_count > 0:
            ordered_weight = ordered_weight.clone()
            ordered_weight[: self.hot_token_count] = ordered_weight[: self.hot_token_count] + self.hot_token_weight.to(dtype=ordered_weight.dtype)
        return _apply_token_permutation(ordered_weight, self.token_permutation)


class UntiedFourierVocab(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 token_order: str = "identity",
                 checkpoint_weight: bool = False,
                 **kwargs):
        super().__init__()
        self.input_vocab = TiedFourierVocab(
            vocab_size,
            hidden_size,
            init_std,
            vocab_modes,
            hidden_modes,
            basis_type=basis_type,
            bias=False,
            embedding_scale=embedding_scale,
            token_order=token_order,
            checkpoint_weight=checkpoint_weight,
            **kwargs,
        )
        self.output_vocab = TiedFourierVocab(
            vocab_size,
            hidden_size,
            init_std,
            vocab_modes,
            hidden_modes,
            basis_type=basis_type,
            bias=bias,
            embedding_scale="none",
            token_order=token_order,
            checkpoint_weight=checkpoint_weight,
            **kwargs,
        )

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        return self.input_vocab.dense_weight(dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.input_vocab.embed(input_ids, weight=weight)

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        return self.output_vocab.logits(hidden_states)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class LearnedTokenFourierVocab(nn.Module, EmbeddingScaleMixin):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 init_std: float,
                 vocab_modes: int,
                 hidden_modes: int,
                 basis_type: str = "dct",
                 bias: bool = False,
                 embedding_scale: str = "init",
                 checkpoint_weight: bool = False,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_modes = min(vocab_modes, vocab_size)
        self.hidden_modes = min(hidden_modes, hidden_size)
        self.basis_type = basis_type
        self.checkpoint_weight = checkpoint_weight

        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}
        coeff_std = init_std * math.sqrt((vocab_size * hidden_size) / (self.vocab_modes * self.hidden_modes))
        token_std = 1.0 / math.sqrt(self.vocab_modes)

        self.token_basis = nn.Parameter(
            trunc_normal_init_(torch.empty((vocab_size, self.vocab_modes), **param_kwargs), std=token_std)  # pyright: ignore[reportArgumentType]
        )
        self.coefficients = nn.Parameter(
            trunc_normal_init_(torch.empty((self.vocab_modes, self.hidden_modes), **param_kwargs), std=coeff_std)  # pyright: ignore[reportArgumentType]
        )
        self.hidden_basis = nn.Buffer(_frequency_basis(hidden_size, self.hidden_modes, basis_type, **basis_kwargs), persistent=False)
        self.bias = nn.Parameter(torch.zeros((vocab_size,), **param_kwargs)) if bias else None
        self._init_embedding_scale(hidden_size, init_std, embedding_scale, **kwargs)

    def _dense_weight_from_parts(self, token_basis: Tensor, coefficients: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        dtype = dtype or coefficients.dtype
        hidden_basis = self.hidden_basis.to(device=coefficients.device, dtype=dtype)
        return token_basis.to(dtype=dtype) @ coefficients.to(dtype=dtype) @ hidden_basis

    def dense_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        if self.checkpoint_weight and self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda token_basis, coefficients: self._dense_weight_from_parts(token_basis, coefficients, dtype=dtype),
                self.token_basis,
                self.coefficients,
                use_reentrant=False,
            )
        return self._dense_weight_from_parts(self.token_basis, self.coefficients, dtype=dtype)

    def embed(self, input_ids: Tensor, weight: Tensor | None = None) -> Tensor:
        dense_weight = self.dense_weight() if weight is None else weight
        return F.embedding(input_ids, dense_weight) * self.embedding_scale

    def logits(self, hidden_states: Tensor, weight: Tensor | None = None) -> Tensor:
        if weight is None and self.checkpoint_weight and self.training and torch.is_grad_enabled():
            def _checkpointed_logits(token_basis, coefficients, hidden_states, bias):
                w = token_basis.to(dtype=hidden_states.dtype) @ coefficients.to(dtype=hidden_states.dtype) @ self.hidden_basis.to(device=coefficients.device, dtype=hidden_states.dtype)
                b = bias.to(dtype=hidden_states.dtype) if bias is not None else None
                return F.linear(hidden_states, w, b)
            
            return checkpoint(
                _checkpointed_logits,
                self.token_basis,
                self.coefficients,
                hidden_states,
                self.bias,
                use_reentrant=False,
            )
        
        dense_weight = self.dense_weight(dtype=hidden_states.dtype) if weight is None else weight.to(dtype=hidden_states.dtype)
        bias = self.bias.to(dtype=hidden_states.dtype) if self.bias is not None else None
        return F.linear(hidden_states, dense_weight, bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)


class LMHead(nn.Module):
    def __init__(self, model: nn.Module, config_dict: dict) -> None:
        super().__init__()
        self.model = model
        # Create cache function
        self.create_cache = self.model.create_cache
        # Train extra args function
        self.compute_train_extra_args = self.model.compute_train_extra_args

        config = LMHeadConfig(**config_dict)
        head_hint: dict = self.model.head_hint  # pyright: ignore[reportAssignmentType]

        input_dim = head_hint["in"]["dim"]
        output_dim = head_hint["out"]["dim"]
        self.vocab_head: nn.Module | None = None

        if config.vocab_head.type != "dense":
            if input_dim != output_dim:
                raise ValueError(f"{config.vocab_head.type} vocab head requires matching input and output hidden dimensions.")
            common_kwargs = dict(
                vocab_size=config.vocab_size,
                hidden_size=input_dim,
                init_std=head_hint["in"]["init_std"],
                bias=config.vocab_head.bias,
                embedding_scale=config.vocab_head.embedding_scale,
            )
            match config.vocab_head.type:
                case "dense_tied":
                    self.vocab_head = DenseTiedVocab(**common_kwargs)
                case "projected_dense_tied":
                    self.vocab_head = ProjectedDenseTiedVocab(**common_kwargs)
                case "tied_lowrank":
                    self.vocab_head = TiedLowRankVocab(
                        **common_kwargs,
                        lowrank_rank=config.vocab_head.lowrank_rank,
                        token_order=config.vocab_head.token_order,
                    )
                case "tiered_hot_lowrank":
                    self.vocab_head = TieredHotLowRankVocab(
                        **common_kwargs,
                        lowrank_rank=config.vocab_head.lowrank_rank,
                        hot_token_count=config.vocab_head.hot_token_count,
                        token_order=config.vocab_head.token_order,
                    )
                case "tied_kronecker":
                    self.vocab_head = TiedKroneckerVocab(
                        **common_kwargs,
                        vocab_factor_a=config.vocab_head.vocab_factor_a,
                        vocab_factor_b=config.vocab_head.vocab_factor_b,
                        hidden_factor_a=config.vocab_head.hidden_factor_a,
                        hidden_factor_b=config.vocab_head.hidden_factor_b,
                        token_order=config.vocab_head.token_order,
                    )
                case "untied_dense":
                    self.vocab_head = UntiedDenseVocab(**common_kwargs)
                case "untied_lowrank":
                    self.vocab_head = UntiedLowRankVocab(
                        **common_kwargs,
                        lowrank_rank=config.vocab_head.lowrank_rank,
                        token_order=config.vocab_head.token_order,
                    )
                case "tied_fourier":
                    self.vocab_head = TiedFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "hybrid_fourier_lowrank":
                    self.vocab_head = HybridFourierLowRankVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        residual_rank=config.vocab_head.residual_rank,
                        residual_scale=config.vocab_head.residual_scale,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "hybrid_fourier_lowrank_asymmetric":
                    self.vocab_head = AsymmetricHybridFourierLowRankVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        residual_rank=config.vocab_head.residual_rank,
                        residual_scale=config.vocab_head.residual_scale,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "factorized_fourier":
                    self.vocab_head = FactorizedFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "multiscale_fourier":
                    self.vocab_head = MultiScaleFourierVocab(
                        **common_kwargs,
                        multiscale_specs=config.vocab_head.multiscale_specs,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                    )
                case "cluster_hybrid_fourier_lowrank":
                    self.vocab_head = ClusterHybridFourierLowRankVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        residual_rank=config.vocab_head.residual_rank,
                        residual_scale=config.vocab_head.residual_scale,
                        num_clusters=config.vocab_head.num_clusters,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "tiered_hot_fourier":
                    self.vocab_head = TieredHotTokenFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        hot_token_count=config.vocab_head.hot_token_count,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "untied_fourier":
                    self.vocab_head = UntiedFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        basis_type=config.vocab_head.basis_type,
                        token_order=config.vocab_head.token_order,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case "learned_token_fourier":
                    self.vocab_head = LearnedTokenFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
                        basis_type=config.vocab_head.basis_type,
                        checkpoint_weight=config.vocab_head.checkpoint_weight,
                    )
                case _:
                    raise NotImplementedError(f"Unsupported vocab_head type: {config.vocab_head.type}")
            self.embed_tokens = self.vocab_head
            self.lm_head = self.vocab_head
        else:
            # LMHead input and output
            self.embed_tokens = ScaledEmbeddingInit(config.vocab_size, input_dim, init_std=head_hint["in"]["init_std"])  # pyright: ignore[reportArgumentType]
            self.lm_head = LinearInit(output_dim, config.vocab_size, bias=False, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]

    def forward(self, carry: Carry, batch: dict[str, Tensor], **kwargs) -> Tuple[Carry, Tensor] | Tuple[Carry, Tensor, dict[str, Tuple[Tensor, Tensor]]]:
        # Token embedding
        tied_weight = None
        if self.vocab_head is not None:
            asymmetric = getattr(self.vocab_head, "asymmetric_embed_logits", False)
            if getattr(self.vocab_head, "checkpoint_weight", False) or asymmetric:
                input_embedding = self.vocab_head.embed(batch["inputs"])
            else:
                tied_weight = self.vocab_head.dense_weight()  # type: ignore[attr-defined]
                input_embedding = self.vocab_head.embed(batch["inputs"], weight=tied_weight)  # type: ignore[attr-defined]
        else:
            input_embedding = self.embed_tokens(batch["inputs"])
            tied_weight = None

        # Model forward
        new_carry, logits = self.model(carry,
                                       input_embedding,
                                       **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                       **kwargs)
        if self.vocab_head is not None:
            asymmetric = getattr(self.vocab_head, "asymmetric_embed_logits", False)
            if getattr(self.vocab_head, "checkpoint_weight", False) or asymmetric:
                logits = self.vocab_head.logits(logits)
            else:
                logits = self.vocab_head.logits(logits, weight=tied_weight)  # type: ignore[attr-defined]
        else:
            logits = self.lm_head(logits)

        # Loss & Metrics
        if "labels" in batch:
            # Masks & labels
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID

            # Loss (CE in F32)
            loss = F.cross_entropy(logits.to(torch.float32), labels.to(torch.long), ignore_index=IGNORE_LABEL_ID, reduction="sum")
            # AllReduce loss divisor. Divide by mean of valid tokens across all processes, as gradient will be averaged.
            loss_divisor = masks.sum().to(torch.float32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)

            # Accuracy
            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                local_valid_counts = masks.sum()
                # Sequence-level statistics
                seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
                seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
                seq_is_valid = seq_num_valid_tokens > 0
                # Metrics
                metrics = {
                    "loss": (loss.detach(), local_valid_counts),
                    "accuracy": (is_correct.sum(), local_valid_counts),
                    "exact_accuracy": (((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(), seq_is_valid.sum()),
                }

            return new_carry, loss / loss_divisor, metrics

        return new_carry, logits
