import torch

import models.layers as layers
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.transformer import TransformerBlock, TransformerConfig


def _delta_attention_class():
    assert hasattr(layers, "DeltaNetAttention"), "DeltaNetAttention is not implemented yet."
    return layers.DeltaNetAttention


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


def test_deltanet_attention_preserves_shape_and_backpropagates():
    DeltaNetAttention = _delta_attention_class()
    torch.manual_seed(41)
    attn = DeltaNetAttention(
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


def test_deltanet_prefix_outputs_do_not_read_response_tokens():
    DeltaNetAttention = _delta_attention_class()
    torch.manual_seed(42)
    prefix_len = 3
    attn = DeltaNetAttention(
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


def test_deltanet_causal_outputs_do_not_read_later_tokens():
    DeltaNetAttention = _delta_attention_class()
    torch.manual_seed(43)
    attn = DeltaNetAttention(
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


def test_preconditioned_deltanet_write_scale_is_positive_and_bounded():
    DeltaNetAttention = _delta_attention_class()
    attn = DeltaNetAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
        preconditioned=True,
        precond_squash=1.5,
    )
    raw = torch.randn(5, 4, 4)

    scale = attn._preconditioner(raw)

    assert scale.shape == raw.shape
    assert torch.all(scale >= 1.0 / 1.5)
    assert torch.all(scale <= 1.5)


def test_transformer_config_can_select_deltanet_token_mixer():
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
        token_mixer="deltanet",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, _delta_attention_class())
    assert not block.attn.preconditioned


def test_transformer_config_can_select_preconditioned_deltanet_token_mixer():
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
        token_mixer="precond_deltanet",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, _delta_attention_class())
    assert block.attn.preconditioned


def test_hrm_can_use_pom_l_level_and_preconditioned_deltanet_h_level():
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
        "H_override": {"token_mixer": "precond_deltanet"},
    }

    model = HierarchicalReasoningModel(config)

    assert isinstance(model.L_level.core.layers[0].attn, layers.PoMAttention)
    assert isinstance(model.H_level.core.layers[0].attn, _delta_attention_class())
    assert model.H_level.core.layers[0].attn.preconditioned
