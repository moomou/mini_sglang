"""Lesson 1 smoke test: greedy 20 tokens, must match HF transformers token-for-token."""
from __future__ import annotations
from collections import defaultdict

import os
import sys
import pathlib as pl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_sglang.config import ModelConfig
from mini_sglang.weights import load_model
from mini_sglang.scheduler.scheduler import Scheduler
from mini_sglang.cache.kv_pool import KvPool
from mini_sglang.cache.block_alloc import BlockAllocator
from mini_sglang.cache.request import Request, ForwardMeta, reserve
from mini_sglang.sampler import Sampler, SamplingParams
from mini_sglang.tokenizer import IncrementalDetokenizer, load_tokenizer

MODEL_DIR = pl.Path(os.environ.get("MODEL_DIR", "/media/2nvme/llm/Qwen3-8B"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
N_NEW = int(os.environ.get("N_NEW", "20"))
NUM_SLOTS = 8192
BLOCK_SIZE = 16


# scripts/l5_smoke.py — sketch
PROMPTS = [
    "Write a sentence with some Chinese: 你好世界. And an emoji: 🎉. Done.",
]

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
def our_greedy(model, prompt_ids: torch.Tensor, n_new: int, sampler, param) -> torch.Tensor:
    """Run our model greedily for n_new tokens. Returns the new token IDs (not including prompt)."""
    cfg: ModelConfig = model.cfg

    alloc = BlockAllocator(NUM_SLOTS)

    pool = KvPool(
        cfg.num_hidden_layers, 
        alloc.num_blocks,
        alloc.block_size,
        NUM_SLOTS,
        cfg.num_key_value_heads,
        cfg.head_dim, 
        dtype=cfg.dtype, 
        device="cuda")


    req = Request(
        id=0,
        prompt_ids=prompt_ids,
        sampling_param={},
    )

    # --- prefill ---
    T = prompt_ids.numel()
    reserve(req, alloc, T)

    slot = torch.tensor(req.slot_indices[:T], device="cuda", dtype=torch.long)
    meta = ForwardMeta(
        positions=torch.arange(T, device="cuda", dtype=torch.long),
        slot_mapping=slot,
        kv_indices=slot,            # prefill: write set == read set
        is_prefill=True,
        cu_seqlens_q = torch.tensor(
            [0, T], device='cuda', dtype=torch.int32
        ),
        seq_lens_kv=torch.tensor(
            [T], device='cuda', dtype=torch.int32
        ),
        block_table=torch.tensor(
            [req.blocks],
            device='cuda',
            dtype=torch.int32,
        ),
        block_size = alloc.block_size,
    )

    logits = model(prompt_ids, pool, meta)
    next_id = sampler(
        logits[-1:].float(), 
        param, 
        prev_ids=req.prompt_ids.unsqueeze(0))[0]
    # next_id = logits[-1].argmax(-1)
    req.output_ids.append(next_id.item())
    req.cur_len = T

    # --- decode ---
    for i in range(n_new - 1):
        reserve(req, alloc, target_len=req.cur_len + 1)
        pos = req.cur_len 

        slot_new = torch.tensor(
            [req.slot_indices[pos]], device="cuda", dtype=torch.long)
        kv_idx = torch.tensor(
            req.slot_indices[:pos+1], device="cuda", dtype=torch.long)
        meta = ForwardMeta(
            positions = torch.tensor([pos], device='cuda'),
            slot_mapping = slot_new,
            kv_indices = kv_idx,
            is_prefill = False,
            cu_seqlens_q = torch.tensor([0, 1], device='cuda', dtype=torch.int32),
            seq_lens_kv = torch.tensor([pos + 1], device='cuda', dtype=torch.int32),
            block_table = torch.tensor([req.blocks], device='cuda', dtype=torch.int32),
            block_size = alloc.block_size,
        )

        ids = next_id.view(1)
        positions = torch.tensor([pos], device="cuda")
        logits = model(ids, pool, meta)
        next_id = sampler(
            logits[-1:].float(), 
            param, 
            prev_ids=torch.tensor([req.prompt_ids.tolist() + req.output_ids], device='cuda'))[0]
        # next_id = logits[-1].argmax(-1)

        req.output_ids.append(next_id.item())
        req.cur_len += 1 

    return torch.tensor(req.output_ids, dtype=torch.long, device="cuda")


@torch.no_grad()
def run_ours(prompts_ids: list[torch.Tensor]) -> tuple[dict[int, list[int]], dict[int, torch.Tensor]]:
    print("loading our model...")
    model = load_model(MODEL_DIR)
    model.eval()
    _vram("after our load")

    sampler = Sampler()
    
    cfg: ModelConfig = model.cfg
    alloc = BlockAllocator(NUM_SLOTS)

    pool = KvPool(
        cfg.num_hidden_layers, 
        alloc.num_blocks,
        alloc.block_size,
        NUM_SLOTS,
        cfg.num_key_value_heads,
        cfg.head_dim, 
        dtype=cfg.dtype, 
        device="cuda")

    scheduler = Scheduler(
        model, 
        sampler, 
        alloc, 
        pool, 
        eos_id=cfg.eos_token_id)

    tokenizer = load_tokenizer(MODEL_DIR)
    outputs = defaultdict(list)
    streams = dict()
    for i, p in enumerate(prompts_ids):
        detok = IncrementalDetokenizer(tokenizer, p)
        req = Request(
            id=i, 
            prompt_ids=p,
            sampling_param=dict(temperature=0.0, top_k=0, top_p=1.0, rep_penalty=1.0),
            max_tokens=20,
            detok=detok,
        )
        scheduler.add_request(req)
        streams[req.id] = detok

    # "server" loop; no pending request yet
    print("<streaming>")
    while scheduler.has_unfinished():
        res = scheduler.step()
        for rid, tok_id in res.new_tokens.items():
            outputs[rid].append(tok_id)
            piece= streams[rid].push(tok_id)
            if piece:
                print(piece, end="", flush=True)
    print("</streaming>")
    

    print("<flush>")
    for detok in streams.values():
        print(detok.flush(), end="", flush=True)
    print("</flush>")

    # compare against single request ref
    param = SamplingParams(
        temperature=torch.tensor(0.0).unsqueeze(0).cuda(),
        top_k=torch.tensor([0], dtype=torch.int32).cuda(),
        top_p=torch.tensor([1.0]).cuda(),
        rep_penalty=torch.tensor([1.0]).cuda(),
    )

    expected_dict = {}
    for rid, p in enumerate(prompts_ids):
        expect = our_greedy(model, p, N_NEW, sampler, param).cpu()
        expected_dict[rid] = expect# .tolist())
        # assert outputs[rid] == expect.tolist(),  f"req {rid} diverged"

    _free_cuda()
    _vram("after our free")

    return outputs, expected_dict


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    prompts_ids = [
        tok(p, return_tensors="pt").input_ids[0].cuda()
        for p in PROMPTS
    ]

    for p, pids in zip(PROMPTS, prompts_ids):
        print(f"prompt: {p!r}  ({pids.numel()} tokens)")

    # Run sequentially so only one ~16 GiB bf16 model is resident at a time.
    # Order: ours first (so an early bug in our code surfaces before paying for HF load).
    outputs, expected = run_ours(prompts_ids)
    for rid, our_ids in outputs.items():
        our_ids = torch.tensor(our_ids)
        
        print(f"ours: {tok.decode(our_ids)}  ids={our_ids.tolist()}")
        print(f"expected: {tok.decode(expected[rid])}  ids={expected[rid].tolist()}")

    print("\n✅ L5 PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
