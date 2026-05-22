from typing import Literal, Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from pydantic import BaseModel, Field

from models.layers import SwiGLU, AttnType, Attention, DeltaNetAttention, FLAGatedDeltaNetAttention, GatedDeltaNetAttention, PolyAttention, PoMAttention, SLAAttention, SpectreAttention, Cache, RotaryEmbedding, find_multiple


class InitConfig(BaseModel):
    in_std: float

    attn_out_std: float
    ff_out_std: float


class FourierLinearConfig(BaseModel):
    enabled: bool = False
    target: Literal["mlp", "attention", "all"] = "mlp"
    in_modes: int = 128
    out_modes: int = 128


class TransformerConfig(BaseModel):
    # Input config
    max_seq_len: int

    # Transformer config
    n_layers: int

    hidden_size: int
    num_heads: int
    expansion: float

    attn_type: AttnType = "prefixlm"

    init_type: Literal["fixed_normal", "lecun_normal", "megatron"]
    init_std: Optional[float] = None

    norm_type: Literal["pre", "post"]
    norm_eps: float

    pos_emb_type: Literal["rope", "none"]
    rope_theta: Optional[float] = None
    token_mixer: Literal["attention", "spectre", "pom", "polyattn", "sla", "deltanet", "precond_deltanet", "gdn", "fla_gdn"] = "attention"
    trm_island_every: int = 0
    trm_island_mixer: Literal["attention", "spectre", "pom", "polyattn", "sla", "deltanet", "precond_deltanet", "gdn"] = "polyattn"
    trm_island_steps: int = 2
    spectre_num_buckets: int = 16
    spectre_gate_hidden: Optional[int] = None
    spectre_dropout: float = 0.0
    pom_order: int = 4
    pom_dropout: float = 0.0
    polyattn_dropout: float = 0.0
    sla_eps: float = 1e-6
    deltanet_eps: float = 1e-6
    precond_squash: float = 1.5
    fourier_linear: FourierLinearConfig = Field(default_factory=FourierLinearConfig)

    # [Computed properties]
    @property
    def intermediate_size(self):
        # Automatic compute "intermediate_size" from "expansion"
        # NOTE: The formula is to match the number of GLU parameters to a vanilla Transformer with same expansion
        return find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
    
    @property
    def init_config(self):
        match self.init_type:
            case "fixed_normal":
                in_std = attn_out_std = ff_out_std = self.init_std if self.init_std is not None else 0.02  # defaults to 0.02, as in OLMo 2
            case "lecun_normal":
                in_std = attn_out_std = 1.0 / math.sqrt(self.hidden_size)
                ff_out_std = 1.0 / math.sqrt(self.intermediate_size)
            case "megatron":
                in_std = self.init_std if self.init_std is not None else 1.0 / math.sqrt(self.hidden_size)
                attn_out_std = ff_out_std = in_std / math.sqrt(2.0 * self.n_layers)
            case _:
                raise NotImplementedError()
            
        return InitConfig(in_std=in_std, attn_out_std=attn_out_std, ff_out_std=ff_out_std)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        attn_kwargs = dict(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            attn_type=config.attn_type,
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.attn_out_std,
            fourier_linear=config.fourier_linear,
        )
        if config.token_mixer == "spectre":
            self.attn = SpectreAttention(
                **attn_kwargs,
                spectre_num_buckets=config.spectre_num_buckets,
                spectre_gate_hidden=config.spectre_gate_hidden,
                spectre_dropout=config.spectre_dropout,
            )
        elif config.token_mixer == "pom":
            self.attn = PoMAttention(
                **attn_kwargs,
                pom_order=config.pom_order,
                pom_dropout=config.pom_dropout,
            )
        elif config.token_mixer == "polyattn":
            self.attn = PolyAttention(
                **attn_kwargs,
                polyattn_dropout=config.polyattn_dropout,
            )
        elif config.token_mixer == "sla":
            self.attn = SLAAttention(
                **attn_kwargs,
                sla_eps=config.sla_eps,
            )
        elif config.token_mixer in {"deltanet", "precond_deltanet"}:
            self.attn = DeltaNetAttention(
                **attn_kwargs,
                preconditioned=config.token_mixer == "precond_deltanet",
                precond_squash=config.precond_squash,
                deltanet_eps=config.deltanet_eps,
            )
        elif config.token_mixer == "gdn":
            self.attn = GatedDeltaNetAttention(
                **attn_kwargs,
                preconditioned=False,
                precond_squash=config.precond_squash,
                deltanet_eps=config.deltanet_eps,
            )
        elif config.token_mixer == "fla_gdn":
            self.attn = FLAGatedDeltaNetAttention(**attn_kwargs)
        else:
            self.attn = Attention(**attn_kwargs)
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            
            init_std_in=config.init_config.in_std,
            init_std_out=config.init_config.ff_out_std,
            fourier_linear=config.fourier_linear,
        )
        
        self.forward = getattr(self, f"_forward_{config.norm_type}")  # Avoid branching logic in "forward" for torch.compile compatibility
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

    # [Forward logic]
    def _forward_pre(self, x: Tensor, **seq_info) -> Tensor:  # Pre Norm
        x = x + self.attn(self.norm(x), **seq_info)
        return x + self.mlp(self.norm(x))
    
    def _forward_post(self, x: Tensor, **seq_info) -> Tensor:  # Post Norm
        x = self.norm(x + self.attn(x, **seq_info))
        return self.norm(x + self.mlp(x))


class TRMIslandBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        island_config = config.model_copy(update={
            "token_mixer": config.trm_island_mixer,
            "trm_island_every": 0,
        })
        self.block = TransformerBlock(island_config)
        self.steps = max(1, int(config.trm_island_steps))

    def forward(self, x: Tensor, cache: Optional[Cache] = None, **seq_info) -> Tensor:
        if cache is not None:
            raise NotImplementedError("TRMIslandBlock does not support cached generation yet.")

        injected = x
        state = x
        for _step in range(self.steps):
            state = self.block(state + injected, **seq_info)
        return state


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_hint = {"in":  {"dim": config.hidden_size, "init_std": config.init_config.in_std},
                          "out": {"dim": config.hidden_size, "init_std": config.init_config.in_std}}  # Hint for LMHead init

        # Position embeddings
        if config.pos_emb_type == "rope":
            assert config.rope_theta is not None
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.max_seq_len, base=config.rope_theta)

        # Layers
        layers = []
        for layer_idx in range(config.n_layers):
            if config.trm_island_every > 0 and (layer_idx + 1) % config.trm_island_every == 0:
                layers.append(TRMIslandBlock(config))
            else:
                layers.append(TransformerBlock(config))
        self.layers = nn.ModuleList(layers)

        # Use final norm only for prenorm
        self.norm_f = lambda x: x
        if config.norm_type == "pre":
            self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1], ), eps=config.norm_eps)

        # Create cache function
        self.create_cache = lambda **kwargs: [Cache.create(**kwargs, num_heads=config.num_heads, head_dim=config.hidden_size // config.num_heads) for _i in range(config.n_layers)]

    def forward(self, x: Tensor, cache: Optional[list[Cache]] = None, **seq_info) -> Tensor:
        seq_info["cos_sin"] = self.rotary_emb(seq_info.pop("position_ids", None)) if hasattr(self, "rotary_emb") else None

        # Forward layers
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, **seq_info, cache=cache[layer_id] if cache is not None else None)

        return self.norm_f(x)
