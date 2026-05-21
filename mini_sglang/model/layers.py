"""Reusable layer primitives. Implement in Lesson 1."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Llama/Qwen-style RMSNorm.

    Reference (HF) computes the normalisation in fp32 and casts back before the weight
    multiplication. Match that for bit-equality with HF greedy decoding.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        in_dtype = x.dtype

        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)

        return self.weight * x32.to(in_dtype)


class Qwen3MLP(nn.Module):
    """SwiGLU MLP: down( silu(gate(x)) * up(x) ). No biases."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def precompute_rope_cache(
    head_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tables of shape [max_position, head_dim]."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    t = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.einsum("p,d->pd", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos().to(dtype), emb.sin().to(dtype)

def _rotate_half(x):
    # x: (T, D)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : ]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
    q: torch.Tensor,  # [T, H_q, D]
    k: torch.Tensor,  # [T, H_kv, D]
    cos: torch.Tensor,  # [T, D]
    sin: torch.Tensor,  # [T, D]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding (rotate-half) to q and k."""
    # q: (T, H_q, D)
    # k: (T, H_kv, D)
    # cos, sin: (T, D) -> (T, 1, D)
    cos = cos.unsqueeze(1)
    sin= sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out
