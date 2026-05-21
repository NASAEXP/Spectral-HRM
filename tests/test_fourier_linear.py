from types import SimpleNamespace

import torch

from models.layers import FourierLinear, LinearInit, SwiGLU
from models.transformer import TransformerBlock, TransformerConfig


def test_fourier_linear_preserves_shape_and_trains_fewer_parameters():
    layer = FourierLinear(
        in_features=16,
        out_features=12,
        bias=True,
        in_modes=4,
        out_modes=5,
    )
    x = torch.randn(3, 7, 16, requires_grad=True)

    y = layer(x)
    y.pow(2).mean().backward()

    dense_parameter_count = 16 * 12 + 12
    fourier_parameter_count = sum(p.numel() for p in layer.parameters())

    assert y.shape == (3, 7, 12)
    assert fourier_parameter_count < dense_parameter_count
    assert layer.coefficients.grad is not None
    assert x.grad is not None


def test_swiglu_can_use_fourier_linears_for_mlp_weights():
    fourier_config = SimpleNamespace(
        enabled=True,
        target="mlp",
        in_modes=4,
        out_modes=5,
    )

    mlp = SwiGLU(
        hidden_size=16,
        intermediate_size=32,
        fourier_linear=fourier_config,
    )

    assert isinstance(mlp.gate_up_proj, FourierLinear)
    assert isinstance(mlp.down_proj, FourierLinear)


def test_transformer_config_keeps_fourier_to_mlp_only_by_default():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        fourier_linear={
            "enabled": True,
            "target": "mlp",
            "in_modes": 4,
            "out_modes": 5,
        },
    )

    block = TransformerBlock(config)

    assert isinstance(block.mlp.gate_up_proj, FourierLinear)
    assert isinstance(block.mlp.down_proj, FourierLinear)
    assert isinstance(block.attn.gqkv_proj, LinearInit)
    assert isinstance(block.attn.o_proj, LinearInit)


def test_transformer_config_can_target_attention_only():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        fourier_linear={
            "enabled": True,
            "target": "attention",
            "in_modes": 4,
            "out_modes": 5,
        },
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn.gqkv_proj, FourierLinear)
    assert isinstance(block.attn.o_proj, FourierLinear)
    assert isinstance(block.mlp.gate_up_proj, LinearInit)
    assert isinstance(block.mlp.down_proj, LinearInit)


def test_transformer_config_can_target_all_projections():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=1,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        fourier_linear={
            "enabled": True,
            "target": "all",
            "in_modes": 4,
            "out_modes": 5,
        },
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn.gqkv_proj, FourierLinear)
    assert isinstance(block.attn.o_proj, FourierLinear)
    assert isinstance(block.mlp.gate_up_proj, FourierLinear)
    assert isinstance(block.mlp.down_proj, FourierLinear)
