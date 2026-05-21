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
    type: Literal["dense", "dense_tied", "tied_fourier", "untied_fourier", "learned_token_fourier"] = "dense"
    vocab_modes: int = 2048
    hidden_modes: int = 512
    basis_type: Literal["dct", "fft"] = "dct"
    bias: bool = False
    embedding_scale: Literal["init", "none", "sqrt_hidden", "learned"] = "init"
    token_order: Literal["identity", "reverse", "even_odd"] = "identity"
    checkpoint_weight: bool = False


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
                case "tied_fourier":
                    self.vocab_head = TiedFourierVocab(
                        **common_kwargs,
                        vocab_modes=config.vocab_head.vocab_modes,
                        hidden_modes=config.vocab_head.hidden_modes,
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
            if getattr(self.vocab_head, "checkpoint_weight", False):
                input_embedding = self.vocab_head.embed(batch["inputs"])
            else:
                tied_weight = self.vocab_head.dense_weight()  # type: ignore[attr-defined]
                input_embedding = self.vocab_head.embed(batch["inputs"], weight=tied_weight)  # type: ignore[attr-defined]
        else:
            input_embedding = self.embed_tokens(batch["inputs"])

        # Model forward
        new_carry, logits = self.model(carry,
                                       input_embedding,
                                       **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                       **kwargs)
        if self.vocab_head is not None:
            if getattr(self.vocab_head, "checkpoint_weight", False):
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
