"""Lesson 1 smoke test: greedy 20 tokens, must match HF transformers token-for-token."""
from __future__ import annotations

import os
import sys
import pathlib as pl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_sglang.config import ModelConfig
from mini_sglang.weights import load_model

MODEL_DIR = pl.Path(os.environ.get("MODEL_DIR", "/media/2nvme/llm/Qwen3-8B"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
N_NEW = int(os.environ.get("N_NEW", "20"))


def _vram(prefix: str) -> None:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        print(f"  [vram] {prefix}: allocated={a:.2f} GiB  reserved={r:.2f} GiB")


def _free_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def our_greedy(model, prompt_ids: torch.Tensor, n_new: int) -> torch.Tensor:
    """Run our model greedily for n_new tokens. Returns the new token IDs (not including prompt)."""
    cfg: ModelConfig = model.cfg
    max_total = prompt_ids.numel() + n_new
    kv_caches = [
        (
            torch.empty(max_total, cfg.num_key_value_heads, cfg.head_dim,
                        dtype=cfg.dtype, device="cuda"),
            torch.empty(max_total, cfg.num_key_value_heads, cfg.head_dim,
                        dtype=cfg.dtype, device="cuda"),
        )
        for _ in range(cfg.num_hidden_layers)
    ]

    # --- prefill ---
    T = prompt_ids.numel()
    positions = torch.arange(T, device="cuda")
    logits = model(prompt_ids, positions, kv_caches, kv_write_offset=0, kv_valid_len=T)
    next_id = logits[-1].argmax(-1)

    out_ids = [next_id]

    # --- decode ---
    for i in range(n_new - 1):
        pos = T + i
        ids = next_id.view(1)
        positions = torch.tensor([pos], device="cuda")
        logits = model(ids, positions, kv_caches,
                       kv_write_offset=pos, kv_valid_len=pos + 1)
        next_id = logits[-1].argmax(-1)
        out_ids.append(next_id)

    return torch.stack(out_ids)


@torch.no_grad()
def run_ours(prompt_ids: torch.Tensor) -> torch.Tensor:
    print("loading our model...")
    model = load_model(MODEL_DIR)
    model.eval()
    _vram("after our load")
    out = our_greedy(model, prompt_ids, N_NEW).cpu()
    del model
    _free_cuda()
    _vram("after our free")
    return out


@torch.no_grad()
def run_hf(prompt_ids: torch.Tensor) -> torch.Tensor:
    print("loading HF reference...")
    hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).cuda().eval()
    _vram("after hf load")
    hf_out = hf.generate(prompt_ids[None], do_sample=False, max_new_tokens=N_NEW)
    out = hf_out[0, prompt_ids.numel():].cpu()
    del hf, hf_out
    _free_cuda()
    _vram("after hf free")
    return out


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids[0].cuda()
    print(f"prompt: {PROMPT!r}  ({prompt_ids.numel()} tokens)")

    # Run sequentially so only one ~16 GiB bf16 model is resident at a time.
    # Order: ours first (so an early bug in our code surfaces before paying for HF load).
    our_ids = run_ours(prompt_ids)
    print(f"ours: {tok.decode(our_ids)}  ids={our_ids.tolist()}")

    hf_ids = run_hf(prompt_ids)
    print(f"hf  : {tok.decode(hf_ids)}  ids={hf_ids.tolist()}")

    if not torch.equal(our_ids, hf_ids):
        print("\n❌ FAIL — token IDs differ")
        # show first divergence position for fast triage
        for i, (a, b) in enumerate(zip(our_ids.tolist(), hf_ids.tolist())):
            if a != b:
                print(f"  first diverge at pos {i}: ours={a} hf={b}")
                break
        return 1
    print("\n✅ L1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
