import sys
import types

import torch

import models.layers as layers
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.transformer import TransformerBlock, TransformerConfig


class FakeFLAGatedDeltaNet(torch.nn.Module):
    last_init_kwargs = None
    last_forward_shape = None

    def __init__(self, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(kwargs["hidden_size"], kwargs["hidden_size"], bias=False)
        FakeFLAGatedDeltaNet.last_init_kwargs = kwargs

    def forward(self, hidden_states, **kwargs):
        FakeFLAGatedDeltaNet.last_forward_shape = tuple(hidden_states.shape)
        return self.proj(hidden_states), None, None


def _install_fake_fla(monkeypatch):
    fla_module = types.ModuleType("fla")
    fla_layers_module = types.ModuleType("fla.layers")
    fla_layers_module.GatedDeltaNet = FakeFLAGatedDeltaNet
    monkeypatch.setitem(sys.modules, "fla", fla_module)
    monkeypatch.setitem(sys.modules, "fla.layers", fla_layers_module)


def _seq_info(*, numseqs: int, seq_len: int, device: torch.device):
    return {
        "cos_sin": None,
        "prefix_lens": torch.full((numseqs,), seq_len // 2, dtype=torch.int32, device=device),
        "causal_lens": torch.full((numseqs,), seq_len - (seq_len // 2), dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(0, (numseqs + 1) * seq_len, seq_len, dtype=torch.int32, device=device),
        "total_seqlen": torch.tensor(numseqs * seq_len, dtype=torch.int64, device=device),
        "numseqs": torch.tensor(numseqs, dtype=torch.int64, device=device),
        "max_seqlen_prefix": torch.tensor(seq_len // 2, dtype=torch.int64, device=device),
        "max_seqlen_causal": torch.tensor(seq_len - (seq_len // 2), dtype=torch.int64, device=device),
        "max_seqlen_all": torch.tensor(seq_len, dtype=torch.int64, device=device),
    }


def test_fla_gdn_attention_uses_optional_fla_layer_and_flattens_back(monkeypatch):
    _install_fake_fla(monkeypatch)
    torch.manual_seed(61)
    attn = layers.FLAGatedDeltaNetAttention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )
    hidden = torch.randn(12, 16, requires_grad=True)

    output = attn(hidden, **_seq_info(numseqs=3, seq_len=4, device=hidden.device))
    output.pow(2).mean().backward()

    assert output.shape == hidden.shape
    assert FakeFLAGatedDeltaNet.last_forward_shape == (3, 4, 16)
    assert hidden.grad is not None
    assert any(param.grad is not None for param in attn.parameters())


def test_fla_gdn_attention_passes_core_shape_knobs(monkeypatch):
    _install_fake_fla(monkeypatch)

    _attn = layers.FLAGatedDeltaNetAttention(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="prefixlm",
    )

    assert FakeFLAGatedDeltaNet.last_init_kwargs["hidden_size"] == 32
    assert FakeFLAGatedDeltaNet.last_init_kwargs["head_dim"] == 8
    assert FakeFLAGatedDeltaNet.last_init_kwargs["num_heads"] == 4
    assert FakeFLAGatedDeltaNet.last_init_kwargs["mode"] == "chunk"
    assert FakeFLAGatedDeltaNet.last_init_kwargs["use_short_conv"] is True


def test_transformer_config_can_select_fla_gdn_token_mixer(monkeypatch):
    _install_fake_fla(monkeypatch)
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
        token_mixer="fla_gdn",
    )

    block = TransformerBlock(config)

    assert isinstance(block.attn, layers.FLAGatedDeltaNetAttention)


def test_hrm_can_use_pom_l_level_and_fla_gdn_h_level(monkeypatch):
    _install_fake_fla(monkeypatch)
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
        "H_override": {"token_mixer": "fla_gdn"},
    }

    model = HierarchicalReasoningModel(config)

    assert isinstance(model.L_level.core.layers[0].attn, layers.PoMAttention)
    assert isinstance(model.H_level.core.layers[0].attn, layers.FLAGatedDeltaNetAttention)
