import sys
import types

import torch

import models.layers as layers
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.transformer import TRMIslandBlock, Transformer, TransformerBlock, TransformerConfig


class FakeFLAGatedDeltaNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(kwargs["hidden_size"], kwargs["hidden_size"], bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states), None, None


def _install_fake_fla(monkeypatch):
    fla_module = types.ModuleType("fla")
    fla_layers_module = types.ModuleType("fla.layers")
    fla_layers_module.GatedDeltaNet = FakeFLAGatedDeltaNet
    monkeypatch.setitem(sys.modules, "fla", fla_module)
    monkeypatch.setitem(sys.modules, "fla.layers", fla_layers_module)


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


def test_poly_attention_preserves_shape_and_backpropagates():
    torch.manual_seed(71)
    attn = layers.PolyAttention(
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


def test_poly_attention_prefix_outputs_do_not_read_response_tokens():
    torch.manual_seed(72)
    prefix_len = 3
    attn = layers.PolyAttention(
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


def test_transformer_can_interleave_polyattn_trm_island():
    config = TransformerConfig(
        max_seq_len=8,
        n_layers=4,
        hidden_size=16,
        num_heads=4,
        expansion=2,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="none",
        token_mixer="pom",
        trm_island_every=4,
        trm_island_mixer="polyattn",
        trm_island_steps=2,
    )

    model = Transformer(config)

    assert isinstance(model.layers[0], TransformerBlock)
    assert isinstance(model.layers[3], TRMIslandBlock)
    assert isinstance(model.layers[3].block.attn, layers.PolyAttention)


def test_hrm_keeps_pom_l_and_fla_h_while_polyattn_only_lives_in_h_island(monkeypatch):
    _install_fake_fla(monkeypatch)
    config = {
        "vocab_size": 260,
        "max_seq_len": 8,
        "n_layers": 8,
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
        "H_override": {
            "token_mixer": "fla_gdn",
            "trm_island_every": 4,
            "trm_island_mixer": "polyattn",
            "trm_island_steps": 2,
        },
    }

    model = HierarchicalReasoningModel(config)

    assert all(isinstance(layer.attn, layers.PoMAttention) for layer in model.L_level.core.layers)
    assert isinstance(model.H_level.core.layers[0].attn, layers.FLAGatedDeltaNetAttention)
    assert isinstance(model.H_level.core.layers[3], TRMIslandBlock)
    assert isinstance(model.H_level.core.layers[3].block.attn, layers.PolyAttention)
