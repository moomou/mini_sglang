"""Qwen3ForCausalLM — eager, single-sequence, contiguous KV. Lesson 1."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_sglang.config import ModelConfig
from mini_sglang.model.layers import Qwen3MLP, RMSNorm
from mini_sglang.model.layers import apply_rope
from mini_sglang.cache.block_alloc import BlockAllocator
from mini_sglang.cache.kv_pool import KvPool
from mini_sglang.cache.request import ForwardMeta

def attn_paged_torch(q, K, V, meta, layer_id, *, H_q, H_kv):
    outputs = []
    num_seqs = meta.cu_seqlens_q.numel() - 1
    for s in range(num_seqs):
        q_start = meta.cu_seqlens_q[s].item()
        q_end= meta.cu_seqlens_q[s+1].item()
        q_len = q_end - q_start
        kv_len = meta.seq_lens_kv[s].item()
        
        # do gather
        n_blocks = (kv_len + meta.block_size - 1) // meta.block_size
        block_ids = meta.block_table[s, :n_blocks]

        K_seq = K[block_ids].reshape(-1, K.shape[-2], K.shape[-1])[:kv_len]
        V_seq = V[block_ids].reshape(-1, V.shape[-2], V.shape[-1])[:kv_len]

        # GQA repeat
        if H_q != H_kv:
            K_seq = K_seq.repeat_interleave(
                H_q // H_kv, dim=1
            )
            V_seq = V_seq.repeat_interleave(
                H_q // H_kv, dim=1
            )

        # SDPA
        # expects batch dim
        Q_s = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        K_s = K_seq.transpose(0, 1).unsqueeze(0)
        V_s = V_seq.transpose(0, 1).unsqueeze(0)
        out_s = F.scaled_dot_product_attention(
            Q_s, K_s, V_s, 
            is_causal=bool(q_len == kv_len),
            dropout_p=0.0,
        )
        outputs.append(out_s.squeeze(0).transpose(0, 1))

    return torch.cat(outputs, dim=0)

class Qwen3Attention(nn.Module):
    """GQA attention with Qwen3 quirks: no bias, q_norm/k_norm on per-head channels,
    RoPE applied AFTER the per-head norm."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x, pool, meta, cos, sin):
        T = x.shape[0]
        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        # project 
        q = self.q_proj(x).view(T, H_q, D)
        k = self.k_proj(x).view(T, H_kv, D)
        v = self.v_proj(x).view(T, H_kv, D)

        # norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # rope q & k
        cos_t = cos[meta.positions]
        sin_t = sin[meta.positions]
        q, k = apply_rope(q, k, cos_t, sin_t)

        # save the cache
        pool.set_kv(self.layer_idx, meta.slot_mapping, k, v)

        # read kv cache
        out= attn_paged_torch(
            q, 
            pool.K[self.layer_idx],
            pool.V[self.layer_idx],
            meta, 
            self.layer_idx, 
            H_q=self.num_heads,
            H_kv=self.num_kv_heads,
        )

        out = out.contiguous().view(T, H_q * D)
        return self.o_proj(out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.mlp = Qwen3MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, pool, meta, cos, sin):
        h = self.input_layernorm(x)
        h = self.self_attn(h, pool, meta, cos, sin)
        x = h + x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = h + x
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids, pool, meta, cos, sin):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, pool, meta, cos, sin)
        return self.norm(x)

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)

        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # rope cache will be filled by load_model() or lazily
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,                    # [T]
        pool: KvPool,
        meta: ForwardMeta,
    ) -> torch.Tensor:                              # [T, vocab]
        h = self.model(
            input_ids, pool, meta,
            self.rope_cos, self.rope_sin,        # buffers attached in load_model()
        )

        return self.lm_head(h)
