"""Static configs (model + server). For Lesson 1 only ModelConfig is used."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int
    bos_token_id: int
    dtype: torch.dtype = torch.bfloat16

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "ModelConfig":
        with open(Path(model_dir) / "config.json") as f:
            c = json.load(f)
        return cls(
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            vocab_size=c["vocab_size"],
            max_position_embeddings=c["max_position_embeddings"],
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            eos_token_id=c["eos_token_id"],
            bos_token_id=c.get("bos_token_id", c["eos_token_id"]),
        )

    @property
    def num_q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
