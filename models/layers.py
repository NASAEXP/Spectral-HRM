from typing import Tuple, Optional, Sequence, Any, NamedTuple, Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from models.common import trunc_normal_init_, unwrap_tensor

try:
    from models import flash_attention_prefixlm_v2 as prefixlm_attention
    flash_attn_varlen_prefixlm = prefixlm_attention.flash_attn_varlen_prefixlm if prefixlm_attention.FLASH_ATTN_AVAILABLE else None
except ImportError:
    flash_attn_varlen_prefixlm = None

try:
    from flash_attn_interface import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


Carry = dict[str, Any]
CosSin = Tuple[Tensor, Tensor]
AttnType = Literal["causal", "prefixlm"]


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x:   [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    # Use FP32 RoPE, as in Transformers OLMo and FlashAttention
    # 
    # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmo/modular_olmo.py#L139-L152
    # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/rotary.h#L126-L133
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base, **kwargs):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor):
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]

        return self.cos_cached, self.sin_cached


class LinearInit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Parameters
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def _cosine_basis(length: int, modes: int, *, device=None, dtype=None) -> Tensor:
    modes = min(modes, length)
    compute_dtype = torch.float32

    positions = torch.arange(length, device=device, dtype=compute_dtype)
    frequencies = torch.arange(modes, device=device, dtype=compute_dtype).unsqueeze(1)
    basis = torch.cos((math.pi / length) * (positions + 0.5) * frequencies)

    basis[0].mul_(1.0 / math.sqrt(length))
    if modes > 1:
        basis[1:].mul_(math.sqrt(2.0 / length))

    return basis.to(dtype=dtype) if dtype is not None else basis


def _config_value(config: Optional[Any], name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _fourier_enabled(config: Optional[Any], role: str) -> bool:
    if not _config_value(config, "enabled", False):
        return False
    target = _config_value(config, "target", "mlp")
    return target == "all" or target == role


def _tensor_item(value: Any) -> int:
    value = unwrap_tensor(value)
    return int(value.item()) if isinstance(value, Tensor) else int(value)


def _torch_prefixlm_attention(q: Tensor,
                              k: Tensor,
                              v: Tensor,
                              is_causal: bool,
                              **seq_info) -> Tensor:
    if q.dim() != 3:
        raise ImportError("flash_attn_interface is required for non-packed Attention forward.")

    total_seqlen = _tensor_item(seq_info.get("total_seqlen", q.shape[0]))
    numseqs = _tensor_item(seq_info.get("numseqs", 1))
    cu_seqlens = unwrap_tensor(seq_info.get("cu_seqlens", torch.tensor([0, total_seqlen], device=q.device)))
    prefix_lens = unwrap_tensor(seq_info.get("prefix_lens", torch.tensor([0], device=q.device)))
    causal_lens = unwrap_tensor(seq_info.get("causal_lens", torch.tensor([total_seqlen], device=q.device)))

    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])

    for seq_idx in range(numseqs):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = end - start
        if seq_len <= 0:
            continue

        q_seq = q[start:end].transpose(0, 1)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)

        scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale
        positions = torch.arange(seq_len, device=q.device)
        if is_causal:
            allowed = positions[:, None] >= positions[None, :]
        else:
            prefix_len = min(int(prefix_lens[seq_idx].item()), seq_len)
            allowed = positions[:, None] >= positions[None, :]
            if prefix_len > 0:
                allowed[:prefix_len, :prefix_len] = True

        scores = scores.masked_fill(~allowed.unsqueeze(0), torch.finfo(scores.dtype).min)
        out[start:end] = torch.matmul(torch.softmax(scores, dim=-1), v_seq).transpose(0, 1)

    if total_seqlen < q.shape[0]:
        out[total_seqlen:] = 0

    return out


def _interp_complex_1d(anchors: Tensor, size: int) -> Tensor:
    if anchors.shape[-1] == size:
        return anchors

    real = F.interpolate(anchors.real.unsqueeze(0), size=size, mode="linear", align_corners=True).squeeze(0)
    imag = F.interpolate(anchors.imag.unsqueeze(0), size=size, mode="linear", align_corners=True).squeeze(0)
    return torch.complex(real, imag)


def _fft_size(seq_len: int) -> int:
    return max(2, 1 << (seq_len - 1).bit_length())


class FourierLinear(nn.Module):
    """Linear layer generated from a compact real Fourier/DCT coefficient grid."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 in_modes: int = 128,
                 out_modes: int = 128,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.total_out_features = math.prod(batch_out_features) * out_features
        self.in_modes = min(in_modes, in_features)
        self.out_modes = min(out_modes, self.total_out_features)

        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        param_kwargs = {k: v for k, v in kwargs.items() if k in ("device", "dtype")}
        basis_kwargs = {k: v for k, v in kwargs.items() if k in ("device",)}

        coeff_std = init_std * math.sqrt((self.total_out_features * in_features) / (self.out_modes * self.in_modes))
        self.coefficients = nn.Parameter(
            trunc_normal_init_(torch.empty((self.out_modes, self.in_modes), **param_kwargs), std=coeff_std)  # pyright: ignore[reportArgumentType]
        )
        self.input_basis = nn.Buffer(_cosine_basis(in_features, self.in_modes, **basis_kwargs), persistent=False)
        self.output_basis = nn.Buffer(_cosine_basis(self.total_out_features, self.out_modes, **basis_kwargs).T, persistent=False)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((self.total_out_features, ), **param_kwargs))

    def forward(self, input: Tensor) -> Tensor:
        input_basis = self.input_basis.to(device=input.device, dtype=input.dtype)
        coefficients = self.coefficients.to(dtype=input.dtype)
        output_basis = self.output_basis.to(device=input.device, dtype=input.dtype)
        bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None

        hidden = F.linear(input, input_basis)
        hidden = F.linear(hidden, coefficients)
        return F.linear(hidden, output_basis, bias)


def make_linear(in_features: int,
                out_features: int,
                bias: bool,
                batch_out_features: Sequence[int] = (),
                init_std: Optional[float] = None,
                fourier_linear: Optional[Any] = None,
                role: str = "mlp",
                **kwargs) -> nn.Module:
    if _fourier_enabled(fourier_linear, role):
        return FourierLinear(
            in_features,
            out_features,
            bias=bias,
            batch_out_features=batch_out_features,
            init_std=init_std,
            in_modes=int(_config_value(fourier_linear, "in_modes", 128)),
            out_modes=int(_config_value(fourier_linear, "out_modes", 128)),
            **kwargs,
        )

    return LinearInit(
        in_features,
        out_features,
        bias=bias,
        batch_out_features=batch_out_features,
        init_std=init_std,
        **kwargs,
    )


class ScaledEmbeddingInit(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 **kwargs):
        super().__init__()
        self.scale = 1.0 / init_std

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


class Cache(NamedTuple):
    """A static cache layer that stores the key and value states as static tensors. Built for `torch.compile` support."""
    keys: Tensor
    values: Tensor

    @classmethod
    def create(cls, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, **kwargs):
        return cls(keys=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs),
                   values=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs))


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, attn_type, init_std_in=None, init_std_out=None, fourier_linear=None, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type

        self.gqkv_proj = make_linear(hidden_size, self.head_dim, batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads, ),
                                     bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="attention", **kwargs)
        self.o_proj = make_linear(head_dim * num_heads, hidden_size,
                                  bias=False, init_std=init_std_out, fourier_linear=fourier_linear, role="attention", **kwargs)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        # hidden_states, gqkv: [..., seq_len, hidden_size]
        gqkv = self.gqkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2 * self.num_heads + 2 * self.num_key_value_heads)
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=-2)
        # query, key, value: [..., seq_len, num_heads, head_dim]
        # RoPE
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        is_causal = self.attn_type == "causal"
        if cache is None:
            # flash attn (training)
            if flash_attn_varlen_prefixlm is None:
                attn_output = _torch_prefixlm_attention(query, key, value, is_causal, **seq_info)
            else:
                attn_output = flash_attn_varlen_prefixlm(query, key, value, is_causal, **{name: unwrap_tensor(tensor) for name, tensor in seq_info.items()})
        else:
            # Regardless of auto / non-autoregressive, apply attention based on current concatenated with cache.
            if flash_attn_with_kvcache is None:
                raise ImportError("flash_attn_interface is required to run cached Attention forward.")
            attn_output = flash_attn_with_kvcache(q=query, k=key, v=value,
                                                  k_cache=cache.keys, v_cache=cache.values, cache_seqlens=cache_lengths,
                                                  num_splits=1,  # Must set to support torch.compile tracing.
                                                  causal=is_causal)  # causal can always be False for PrefixLM. during AR generation seqlen is 1, so causal masking won't matter.

        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(torch.sigmoid(gate) * attn_output, "... h hd -> ... (h hd)")  # type: ignore
        return self.o_proj(attn_output)


class PoMAttention(nn.Module):
    """PoM-style polynomial token mixer with PrefixLM-safe masking.

    This keeps the same call shape as Attention so it can act as the fast
    L-level mixer. It mixes tokens through cumulative polynomial moments
    instead of pairwise attention scores.
    """

    def __init__(self,
                 hidden_size,
                 head_dim,
                 num_heads,
                 num_key_value_heads,
                 attn_type,
                 init_std_in=None,
                 init_std_out=None,
                 fourier_linear=None,
                 pom_order: int = 4,
                 pom_dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        if hidden_size != head_dim * num_heads:
            raise ValueError("PoMAttention requires hidden_size == head_dim * num_heads.")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        self.order = max(1, int(pom_order))

        self.v_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="attention", **kwargs)
        self.gate_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="attention", **kwargs)
        self.dropout = nn.Dropout(pom_dropout) if pom_dropout > 0 else nn.Identity()
        self.o_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_out, fourier_linear=fourier_linear, role="attention", **kwargs)

    def _basis(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len == 1:
            pos = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=torch.float32)
        return torch.stack([pos.pow(order) for order in range(self.order)], dim=-1)

    def _mix_projected_sequence(self, v_seq: Tensor, prefix_len: int, output_dtype: torch.dtype) -> Tensor:
        seq_len = v_seq.shape[0]
        basis = self._basis(seq_len, device=v_seq.device)
        values = v_seq.float()
        weighted = basis.unsqueeze(-1) * values.unsqueeze(1)

        denom = basis.cumsum(dim=0).clamp_min(1e-6)
        summary = weighted.cumsum(dim=0) / denom.unsqueeze(-1)
        mixed = (summary * basis.unsqueeze(-1)).sum(dim=1)

        if self.attn_type == "prefixlm" and prefix_len > 0:
            prefix_basis = basis[:prefix_len]
            prefix_summary = weighted[:prefix_len].sum(dim=0) / prefix_basis.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)
            mixed[:prefix_len] = (prefix_summary.unsqueeze(0) * prefix_basis.unsqueeze(-1)).sum(dim=1)

        return mixed.to(output_dtype)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        if cache is not None:
            raise NotImplementedError("PoMAttention does not support cached generation yet.")
        if hidden_states.dim() != 2:
            raise NotImplementedError("PoMAttention currently supports packed [tokens, hidden] inputs only.")

        total_seqlen = _tensor_item(seq_info.get("total_seqlen", hidden_states.shape[0]))
        numseqs = _tensor_item(seq_info.get("numseqs", 1))
        cu_seqlens = unwrap_tensor(seq_info.get("cu_seqlens", torch.tensor([0, total_seqlen], device=hidden_states.device)))
        prefix_lens = unwrap_tensor(seq_info.get("prefix_lens", torch.tensor([0], device=hidden_states.device)))

        out = torch.zeros_like(hidden_states)
        values = self.v_proj(hidden_states)
        gates = torch.sigmoid(self.gate_proj(hidden_states))

        for seq_idx in range(numseqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue

            prefix_len = min(int(prefix_lens[seq_idx].item()), seq_len) if self.attn_type == "prefixlm" else 0
            mixed = self._mix_projected_sequence(values[start:end], prefix_len, hidden_states.dtype)
            out[start:end] = self.dropout(gates[start:end] * mixed)

        if total_seqlen < hidden_states.shape[0]:
            out[total_seqlen:] = 0

        return self.o_proj(out)


class SpectreAttention(nn.Module):
    """SPECTRE-style FFT token mixer with PrefixLM-safe local masking.

    This is intentionally simple and slow for now. It proves the mixer can be
    wired into HRM-Text without leaking future response tokens.
    """

    def __init__(self,
                 hidden_size,
                 head_dim,
                 num_heads,
                 num_key_value_heads,
                 attn_type,
                 init_std_in=None,
                 init_std_out=None,
                 fourier_linear=None,
                 spectre_num_buckets: int = 16,
                 spectre_gate_hidden: Optional[int] = None,
                 spectre_dropout: float = 0.0,
                 **kwargs):
        super().__init__()
        if hidden_size != head_dim * num_heads:
            raise ValueError("SpectreAttention requires hidden_size == head_dim * num_heads.")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        self.num_buckets = max(4, int(spectre_num_buckets))

        gate_hidden = spectre_gate_hidden or hidden_size
        self.q_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="attention", **kwargs)
        self.v_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="attention", **kwargs)
        self.gate_norm = nn.LayerNorm(hidden_size)
        self.gate_in = LinearInit(hidden_size, gate_hidden, bias=True, init_std=init_std_in, **kwargs)
        self.gate_out = LinearInit(gate_hidden, num_heads * self.num_buckets * 2, bias=True, init_std=init_std_in, **kwargs)
        self.modrelu_bias = nn.Parameter(torch.full((num_heads, 1), -0.1, **{k: v for k, v in kwargs.items() if k in ("device", "dtype")}))
        self.dropout = nn.Dropout(spectre_dropout) if spectre_dropout > 0 else nn.Identity()
        self.o_proj = make_linear(hidden_size, hidden_size, bias=False, init_std=init_std_out, fourier_linear=fourier_linear, role="attention", **kwargs)

        with torch.no_grad():
            if self.gate_out.bias is not None:
                self.gate_out.bias.zero_()
                gate_bias = self.gate_out.bias.view(num_heads, self.num_buckets, 2)
                gate_bias[..., 0].fill_(1.0)

    def _complex_gate(self, q_context: Tensor, spectrum_size: int) -> Tensor:
        q_pool = self.gate_norm(q_context.mean(dim=0))
        anchors = self.gate_out(F.gelu(self.gate_in(q_pool))).view(self.num_heads, self.num_buckets, 2)
        gate = torch.view_as_complex(anchors.float())
        gate = _interp_complex_1d(gate, spectrum_size)

        magnitude = torch.abs(gate)
        bias = self.modrelu_bias.float()
        scale = F.relu(magnitude + bias) / torch.sqrt(magnitude.square() + 1e-8)
        return gate * scale

    def _mix_projected_context(self, q_context: Tensor, v_context: Tensor, output_dtype: torch.dtype) -> Tensor:
        seq_len = q_context.shape[0]
        n_fft = _fft_size(seq_len)

        v = rearrange(v_context, "s (h hd) -> s h hd", h=self.num_heads).float()

        v_fft = torch.fft.rfft(v, n=n_fft, dim=0)
        gate = self._complex_gate(q_context, v_fft.shape[0])
        mixed_fft = v_fft * gate.T.unsqueeze(-1)
        mixed = torch.fft.irfft(mixed_fft, n=n_fft, dim=0)[:seq_len]
        mixed = rearrange(mixed.to(output_dtype), "s h hd -> s (h hd)")
        return self.dropout(mixed)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        if cache is not None:
            raise NotImplementedError("SpectreAttention does not support cached generation yet.")
        if hidden_states.dim() != 2:
            raise NotImplementedError("SpectreAttention currently supports packed [tokens, hidden] inputs only.")

        total_seqlen = _tensor_item(seq_info.get("total_seqlen", hidden_states.shape[0]))
        numseqs = _tensor_item(seq_info.get("numseqs", 1))
        cu_seqlens = unwrap_tensor(seq_info.get("cu_seqlens", torch.tensor([0, total_seqlen], device=hidden_states.device)))
        prefix_lens = unwrap_tensor(seq_info.get("prefix_lens", torch.tensor([0], device=hidden_states.device)))

        out = torch.zeros_like(hidden_states)
        for seq_idx in range(numseqs):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = end - start
            if seq_len <= 0:
                continue

            seq = hidden_states[start:end]
            prefix_len = min(int(prefix_lens[seq_idx].item()), seq_len) if self.attn_type == "prefixlm" else 0
            q_seq = self.q_proj(seq)
            v_seq = self.v_proj(seq)

            if self.attn_type == "prefixlm" and prefix_len > 0:
                out[start:start + prefix_len] = self._mix_projected_context(q_seq[:prefix_len], v_seq[:prefix_len], seq.dtype)

            for pos in range(prefix_len, seq_len):
                out[start + pos] = self._mix_projected_context(q_seq[:pos + 1], v_seq[:pos + 1], seq.dtype)[-1]

        if total_seqlen < hidden_states.shape[0]:
            out[total_seqlen:] = 0

        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std_in=None, init_std_out=None, fourier_linear=None, **kwargs):
        super().__init__()
        self.gate_up_proj = make_linear(hidden_size, intermediate_size, batch_out_features=(2, ),
                                        bias=False, init_std=init_std_in, fourier_linear=fourier_linear, role="mlp", **kwargs)
        self.down_proj    = make_linear(intermediate_size, hidden_size,
                                        bias=False, init_std=init_std_out, fourier_linear=fourier_linear, role="mlp", **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
