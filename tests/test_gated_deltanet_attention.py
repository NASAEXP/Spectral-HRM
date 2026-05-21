import torch

import models.layers as layers
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.transformer import TransformerBlock, TransformerConfig


def _gdn_attention_class():
    assert hasattr(layers, "GatedDeltaNetAttention"), "GatedDeltaNetAttention is not implemented yet."
    return layers.GatedDeltaNetAttention


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


def test_gdn_attention_preserves_shape_and_backpropagates():
    GatedDeltaNetAttention = _gdn_attention_class()
    torch.manual_seed(51)
    attn = GatedDeltaNetAttention(
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


def test_gdn_prefix_outputs_do_not_read_response_tokens():
    GatedDeltaNetAttention = _gdn_attention_class()
    torch.manual_seed(52)
    prefix_len = 3
    attn = GatedDeltaNetAttention(
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


def test_gdn_causal_outputs_do_not_read_later_tokens():
    GatedDeltaNetAttention = _gdn_attention_class()
    torch.manual_seed(53)
    attn = GatedDeltaNetAttention(
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


def test_gdn_update_scales_erasure_by_decay_gate():
    GatedDeltaNetAttention = _gdn_attention_class()
    attn = GatedDeltaNetAttention(
        hidden_size=2,
        head_dim=2,
        num_heads=1,
        num_key_value_heads=1,
        attn_type="causal",
    )
    state = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]])
    key = torch.tensor([[1.0, 0.0]])
    value = torch.tensor([[3.0, 0.0]])
    beta = torch.tensor([0.5])
    decay = torch.tensor([0.25])

    updated = attn._update_state(state, key, value, beta, decay)

    expected = torch.tensor([[[1.75, 0.0], [0.0, 0.0]]])
    assert torch.allclose(updated, expected)


def test_transformer_config_can_select_gdn_token_mixer():
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
        token_mixer="gdn",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, _gdn_attention_class())


def test_hrm_can_use_pom_l_level_and_gdn_h_level():
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
        "H_override": {"token_mixer": "gdn"},
    }

    model = HierarchicalReasoningModel(config)

    assert isinstance(model.L_level.core.layers[0].attn, layers.PoMAttention)
    assert isinstance(model.H_level.core.layers[0].attn, _gdn_attention_class())
