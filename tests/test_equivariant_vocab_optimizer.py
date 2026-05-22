"""Tests for mixed symmetry vocab optimizers."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from equivariant_vocab_optimizer import MultiOptimizer, build_training_optimizer, partition_vocab_matrix_params


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        self.vocab_head = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.vocab_head.T + self.body(x)


def test_partition_vocab_matrix_params():
    model = _TinyModel()
    vocab, other = partition_vocab_matrix_params(model)
    assert len(vocab) == 1
    assert vocab[0] is model.vocab_head
    assert len(other) == 2


def test_build_adamw_only():
    model = _TinyModel()
    opt = build_training_optimizer(model, vocab_optimizer="adamw")
    assert isinstance(opt, (torch.optim.AdamW, MultiOptimizer))


def test_build_rownorm_mixed():
    equiv_root = REPO_ROOT / ".tools" / "equivariant_optimizers"
    if not equiv_root.is_dir():
        return
    model = _TinyModel()
    opt = build_training_optimizer(model, vocab_optimizer="rownorm")
    model.vocab_head.grad = torch.randn_like(model.vocab_head)
    model.body.weight.grad = torch.randn_like(model.body.weight)
    opt.zero_grad()
    opt.step()


def test_build_rightpolar_step():
    equiv_root = REPO_ROOT / ".tools" / "equivariant_optimizers"
    if not equiv_root.is_dir():
        return
    model = _TinyModel()
    opt = build_training_optimizer(model, vocab_optimizer="rightpolar", polar_num_steps=2)
    model.vocab_head.grad = torch.randn_like(model.vocab_head)
    opt.zero_grad()
    opt.step()
