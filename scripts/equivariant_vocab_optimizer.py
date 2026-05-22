"""Mixed AdamW + symmetry-compatible optimizers for vocab/LM-head matrices."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

EQUIV_ROOT = Path(__file__).resolve().parents[1] / ".tools" / "equivariant_optimizers"


def _import_equivariant():
    root = str(EQUIV_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from optim import HybridPolarGradM, RightPolarGradM, RowNormM

    return RowNormM, RightPolarGradM, HybridPolarGradM


VocabOptimizerName = Literal["adamw", "rownorm", "rightpolar", "hybrid"]


class MultiOptimizer:
    """Step multiple optimizers; zero_grad/step all."""

    def __init__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()


class PolarEveryK(Optimizer):
    """Run polar optimizer every k steps; AdamW on other steps (same params)."""

    def __init__(self, params, *, polar: Optimizer, adam: Optimizer, every_k: int) -> None:
        if every_k < 1:
            raise ValueError("every_k must be >= 1.")
        self.polar = polar
        self.adam = adam
        self.every_k = every_k
        self._step_idx = 0
        super().__init__(params, defaults={})

    @torch.no_grad()
    def step(self, closure=None) -> None:
        del closure
        self._step_idx += 1
        if self._step_idx % self.every_k == 0:
            self.polar.step()
        else:
            self.adam.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.polar.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)


def partition_vocab_matrix_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """2D vocab_head weights (embed/LM matrix), vs everything else."""
    vocab_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vocab_head" in name and param.ndim == 2:
            vocab_params.append(param)
        else:
            other_params.append(param)
    return vocab_params, other_params


def build_training_optimizer(
    model: nn.Module,
    *,
    vocab_optimizer: VocabOptimizerName = "adamw",
    lr: float = 2e-3,
    lr_vocab: float | None = None,
    weight_decay: float = 0.01,
    beta: float = 0.95,
    polar_num_steps: int = 5,
    polar_alpha: float = 1.0,
    polar_backend: str = "polar_express",
    polar_every: int = 1,
    hybrid_alpha: float = 1.0,
) -> Optimizer:
    """
    AdamW on the body; symmetry-aware optimizer on 2D vocab_head matrices only.

    Note: RightPolarGradM / HybridPolarGradM apply polar decomp to G^T G with shape
    (hidden, hidden) — for (vocab, hidden) that is 256x256 here, not vocab x vocab.
    """
    lr_vocab = lr if lr_vocab is None else lr_vocab
    vocab_params, other_params = partition_vocab_matrix_params(model)
    optimizers: list[Optimizer] = []

    if other_params:
        optimizers.append(torch.optim.AdamW(other_params, lr=lr, weight_decay=weight_decay))

    if vocab_params:
        if vocab_optimizer == "adamw":
            optimizers.append(torch.optim.AdamW(vocab_params, lr=lr_vocab, weight_decay=weight_decay))
        else:
            RowNormM, RightPolarGradM, HybridPolarGradM = _import_equivariant()
            if vocab_optimizer == "rownorm":
                vocab_opt: Optimizer = RowNormM(
                    vocab_params,
                    lr=lr_vocab,
                    beta=beta,
                    weight_decay=weight_decay,
                    orientation="row",
                )
            elif vocab_optimizer == "rightpolar":
                polar_opt = RightPolarGradM(
                    vocab_params,
                    lr=lr_vocab,
                    beta=beta,
                    alpha=polar_alpha,
                    weight_decay=weight_decay,
                    backend=polar_backend,
                    num_steps=polar_num_steps,
                )
                if polar_every > 1:
                    adam_opt = torch.optim.AdamW(vocab_params, lr=lr_vocab, weight_decay=weight_decay)
                    vocab_opt = PolarEveryK(vocab_params, polar=polar_opt, adam=adam_opt, every_k=polar_every)
                else:
                    vocab_opt = polar_opt
            elif vocab_optimizer == "hybrid":
                vocab_opt = HybridPolarGradM(
                    vocab_params,
                    lr=lr_vocab,
                    beta=beta,
                    alpha=hybrid_alpha,
                    weight_decay=weight_decay,
                    backend=polar_backend,
                    num_steps=polar_num_steps,
                    orientation="row",
                )
            else:
                raise ValueError(f"Unknown vocab_optimizer: {vocab_optimizer}")
            optimizers.append(vocab_opt)

    if not optimizers:
        raise ValueError("No trainable parameters found.")
    if len(optimizers) == 1:
        return optimizers[0]
    return MultiOptimizer(optimizers)
