import torch
from torch import nn

import models.layers as layers
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.transformer import TransformerBlock, TransformerConfig


def _spectre_attention_class():
    assert hasattr(layers, "SpectreAttention"), "SpectreAttention is not implemented yet."
    return layers.SpectreAttention


def _pom_attention_class():
    assert hasattr(layers, "PoMAttention"), "PoMAttention is not implemented yet."
    return layers.PoMAttention


def _seq_info(*, prefix_len: int, causal_len: int, device: torch.device):
    total_len = prefix_len + causal_len
    return {
        "cos_sin": None,
        "prefix_lens": torch.tensor([prefix_len], dtype=torch.int32, device=device),
        "causal_lens": torch.tensor([causal_len], dtype=torch.int32, device=device),
        "cu_seqlens": torch.tensor([0, total_len], dtype=torch.int32, device=device),
        "total_seqlen": torch.tensor(total_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(1, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(prefix_len, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(causal_len, dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(total_len, dtype=torch.int64, device=device),
    }


class _CountingProjection(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.wrapped(x)


def test_spectre_attention_preserves_shape_and_backpropagates():
    SpectreAttention = _spectre_attention_class()
    torch.manual_seed(1)
    attn = SpectreAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    hidden = torch.randn(7, 16, requires_grad=True)

    output = attn(hidden, **_seq_info(prefix_len=3, causal_len=4, device=hidden.device))
    output.pow(2).mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert any(param.grad is not None for param in attn.parameters())


def test_spectre_attention_precomputes_projections_once_per_sequence():
    SpectreAttention = _spectre_attention_class()
    torch.manual_seed(4)
    attn = SpectreAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    attn.q_proj = _CountingProjection(attn.q_proj)
    attn.v_proj = _CountingProjection(attn.v_proj)
    hidden = torch.randn(7, 16)

    attn(hidden, **_seq_info(prefix_len=3, causal_len=4, device=hidden.device))

    assert attn.q_proj.calls == 1
    assert attn.v_proj.calls == 1


def test_spectre_attention_handles_bfloat16_forward_params():
    SpectreAttention = _spectre_attention_class()
    torch.manual_seed(11)
    attn = SpectreAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    ).to(dtype=torch.bfloat16)
    hidden = torch.randn(7, 16, dtype=torch.bfloat16)

    output = attn(hidden, **_seq_info(prefix_len=3, causal_len=4, device=hidden.device))

    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16


def test_spectre_prefix_outputs_do_not_read_response_tokens():
    SpectreAttention = _spectre_attention_class()
    torch.manual_seed(2)
    prefix_len = 3
    attn = SpectreAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    hidden = torch.randn(7, 16)
    changed_future = hidden.clone()
    changed_future[prefix_len:] = torch.randn_like(changed_future[prefix_len:]) * 10.0

    seq_info = _seq_info(prefix_len=prefix_len, causal_len=4, device=hidden.device)
    base = attn(hidden, **seq_info)
    changed = attn(changed_future, **seq_info)

    assert torch.allclose(base[:prefix_len], changed[:prefix_len], atol=1e-5, rtol=1e-5)


def test_spectre_causal_outputs_do_not_read_later_tokens():
    SpectreAttention = _spectre_attention_class()
    torch.manual_seed(3)
    attn = SpectreAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="causal",
    )
    hidden = torch.randn(6, 16)
    changed_future = hidden.clone()
    changed_future[4:] = torch.randn_like(changed_future[4:]) * 10.0

    seq_info = _seq_info(prefix_len=0, causal_len=6, device=hidden.device)
    base = attn(hidden, **seq_info)
    changed = attn(changed_future, **seq_info)

    assert torch.allclose(base[:4], changed[:4], atol=1e-5, rtol=1e-5)


def test_transformer_config_can_select_spectre_token_mixer():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        token_mixer="spectre",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, _spectre_attention_class())


def test_pom_attention_preserves_shape_and_backpropagates():
    PoMAttention = _pom_attention_class()
    torch.manual_seed(21)
    attn = PoMAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    hidden = torch.randn(7, 16, requires_grad=True)

    output = attn(hidden, **_seq_info(prefix_len=3, causal_len=4, device=hidden.device))
    output.pow(2).mean().backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert any(param.grad is not None for param in attn.parameters())


def test_pom_prefix_outputs_do_not_read_response_tokens():
    PoMAttention = _pom_attention_class()
    torch.manual_seed(22)
    prefix_len = 3
    attn = PoMAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    hidden = torch.randn(7, 16)
    changed_future = hidden.clone()
    changed_future[prefix_len:] = torch.randn_like(changed_future[prefix_len:]) * 10.0

    seq_info = _seq_info(prefix_len=prefix_len, causal_len=4, device=hidden.device)
    base = attn(hidden, **seq_info)
    changed = attn(changed_future, **seq_info)

    assert torch.allclose(base[:prefix_len], changed[:prefix_len], atol=1e-5, rtol=1e-5)


def test_pom_causal_outputs_do_not_read_later_tokens():
    PoMAttention = _pom_attention_class()
    torch.manual_seed(23)
    attn = PoMAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="causal",
    )
    hidden = torch.randn(6, 16)
    changed_future = hidden.clone()
    changed_future[4:] = torch.randn_like(changed_future[4:]) * 10.0

    seq_info = _seq_info(prefix_len=0, causal_len=6, device=hidden.device)
    base = attn(hidden, **seq_info)
    changed = attn(changed_future, **seq_info)

    assert torch.allclose(base[:4], changed[:4], atol=1e-5, rtol=1e-5)


def test_transformer_config_can_select_pom_token_mixer():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        token_mixer="pom",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, _pom_attention_class())


def test_hrm_can_use_pom_l_level_and_spectre_h_level():
    config = {
        "vocab_size": 260,
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
        "token_mixer": "pom",
        "H_override": {"token_mixer": "spectre"},
    }

    model = HierarchicalReasoningModel(config)

    assert isinstance(model.L_level.core.layers[0].attn, _pom_attention_class())
    assert isinstance(model.H_level.core.layers[0].attn, _spectre_attention_class())
