"""Safetensors weight loader. Implement in Lesson 1."""
from __future__ import annotations

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import safetensors
from safetensors.torch import load_file
from mini_sglang.config import ModelConfig
from mini_sglang.model.qwen3 import Qwen3ForCausalLM
from mini_sglang.model.layers import precompute_rope_cache


def load_model(model_dir: str | Path) -> nn.Module:
    """Build Qwen3ForCausalLM and load all safetensors shards into it.

    Lesson 1 task: implement this.

    Suggested approach:
      1. ModelConfig.from_pretrained(model_dir) -> cfg
      2. model = Qwen3ForCausalLM(cfg).to(cfg.dtype).cuda()
      3. Read model.safetensors.index.json -> weight_map: dict[name -> shard_filename]
      4. For each unique shard, open with safetensors.safe_open(..., device='cuda')
         and copy tensors into the matching parameters of `model`.
      5. Verify every parameter got loaded (track a set of loaded names).

    Return the model in eval() mode.
    """
    cfg = ModelConfig.from_pretrained(model_dir)
    model = Qwen3ForCausalLM(cfg).to(cfg.dtype).cuda()

    with open(model_dir / "model.safetensors.index.json", 'r') as f:
      index = json.load(f)

    # group layer names by safetensor files
    grouped = {}
    for k,v in index['weight_map'].items():
      grouped.setdefault(v, []).append(k)

    # now load each safetensor
    state = model.state_dict()
    loaded = set()
    for tpath in grouped.keys():
      tensors = load_file(model_dir / tpath, device="cuda")
      loaded = loaded | set(tensors.keys())
      model.load_state_dict(tensors, strict=False)

    if (missing := set(state.keys()) - loaded):
      raise f"invalid load:: {missing}"

    cos, sin = precompute_rope_cache(cfg.head_dim, cfg.max_position_embeddings,
                                cfg.rope_theta, dtype=cfg.dtype, device="cuda")
    model.register_buffer("rope_cos", cos, persistent=False)
    model.register_buffer("rope_sin", sin, persistent=False)

    return model.eval()