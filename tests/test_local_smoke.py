import torch

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead


def _tiny_batch(vocab_size: int) -> dict[str, torch.Tensor]:
    prefix_lens = torch.tensor([3, 2], dtype=torch.int32)
    causal_lens = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

    return {
        "inputs": torch.randint(1, vocab_size, (8,), dtype=torch.int64),
        "labels": torch.randint(1, vocab_size, (8,), dtype=torch.int64),
        "position_ids": torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int64),
        "prefix_lens": prefix_lens,
        "causal_lens": causal_lens,
        "cu_seqlens": cu_seqlens,
        "total_seqlen": torch.tensor(8, dtype=torch.int64),
        "numseqs": torch.tensor(2, dtype=torch.int64),
        "max_seqlen_prefix": torch.tensor(3, dtype=torch.int64),
        "max_seqlen_causal": torch.tensor(2, dtype=torch.int64),
        "max_seqlen_all": torch.tensor(5, dtype=torch.int64),
    }


def test_tiny_fourier_hrm_text_step_runs_without_flash_or_distributed():
    vocab_size = 64
    config = {
        "vocab_size": vocab_size,
        "max_seq_len": 8,
        "n_layers": 2,
        "hidden_size": 16,
        "num_heads": 4,
        "expansion": 2,
        "attn_type": "prefixlm",
        "init_type": "lecun_normal",
        "norm_type": "pre",
        "norm_eps": 1e-6,
        "pos_emb_type": "none",
        "half_layers": True,
        "H_cycles": 1,
        "L_cycles": 1,
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 2,
        "bp_max_steps": 2,
        "H_override": {},
        "fourier_linear": {
            "enabled": True,
            "target": "mlp",
            "in_modes": 4,
            "out_modes": 5,
        },
    }
    model = LMHead(HierarchicalReasoningModel(config), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _carry, loss, metrics = model(carry=None, batch=_tiny_batch(vocab_size), bp_steps=2)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert metrics["loss"][1].item() == 8
