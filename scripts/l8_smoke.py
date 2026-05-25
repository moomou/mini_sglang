"""Lesson 8 smoke test — radix prefix cache.

Three phases:
  A. Unit tests of RadixCache + BlockAllocator refcount semantics (no model).
  B. End-to-end with model: prime the cache with a prompt, then send a request
     that shares the prefix. Show prefill compute drops.
  C. Different prompt → no cache hit.

Acceptance:
  - A: all assertions pass.
  - B: total prefill q tokens for the cached run < the no-cache baseline.
  - B: cached output equals baseline output bit-for-bit (cache is correct).
  - C: unrelated prompt yields matched_len == 0.

Run unit tests only (fast, no GPU model load):
  L8_SKIP_MODEL=1 python -m scripts.l8_smoke
"""
from __future__ import annotations

import os
import sys
import time
import pathlib as pl

import torch
from transformers import AutoTokenizer

from mini_sglang.config import ModelConfig
from mini_sglang.weights import load_model
from mini_sglang.scheduler.scheduler import Scheduler
from mini_sglang.cache.kv_pool import KvPool
from mini_sglang.cache.block_alloc import BlockAllocator
from mini_sglang.cache.request import Request
from mini_sglang.cache.radix_cache import RadixCache
from mini_sglang.sampler import Sampler

MODEL_DIR = pl.Path(os.environ.get("MODEL_DIR", "/media/2nvme/llm/Qwen3-8B"))
NUM_SLOTS = 8192
N_NEW     = int(os.environ.get("N_NEW", "20"))


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — unit tests (no model)
# ─────────────────────────────────────────────────────────────────────────────

def phase_a_unit_tests() -> None:
    print("\n=== Phase A: RadixCache unit tests ===")

    alloc = BlockAllocator(32, block_size=2)
    c     = RadixCache(alloc)

    # A.1 empty cache
    blocks, m = c.match([1, 2, 3, 4])
    assert (blocks, m) == ([], 0), f"empty match wrong: {(blocks, m)}"
    print("  A.1 empty cache match     ✓")

    # A.2 insert + exact match. Convention: caller already alloc'd blocks 5,7
    # (refcount=1 each); insert increfs → both at refcount=2.
    for b in (5, 7):
        alloc.refcount[b] = 1
    c.insert([1, 2, 3, 4], [5, 7])
    assert alloc.refcount[5] == 2 and alloc.refcount[7] == 2, \
        f"refcounts after insert: 5={alloc.refcount[5]}, 7={alloc.refcount[7]}"
    blocks, m = c.match([1, 2, 3, 4])             # match increfs again → 3
    assert (blocks, m) == ([5, 7], 4), f"exact match wrong: {(blocks, m)}"
    alloc.decref(blocks)                          # release the match incref
    print("  A.2 exact match           ✓")

    # A.3 partial match at odd-token boundary (regression test for the rounding bug)
    blocks, m = c.match([1, 2, 3, 99])
    assert (blocks, m) == ([5], 2), f"partial match should block-align down: {(blocks, m)}"
    alloc.decref(blocks)
    print("  A.3 partial match block-aligned ✓")

    # A.4 insert that forces a split (shares [1,2] prefix with prior insert)
    time.sleep(0.005)                              # distinct last_access_time
    for b in (9, 11):
        alloc.refcount[b] = 1
    c.insert([1, 2, 5, 6], [9, 11])
    # NOTE: block 9 covered [1,2] which the cache already had as block 5.
    # Cache silently dropped block 9 (it's not in the tree). Caller's responsibility
    # to decref it after insert. Document this contract in your integration.
    alloc.decref([9])

    for prompt, want in [
        ([1, 2, 3, 4], ([5, 7], 4)),
        ([1, 2, 5, 6], ([5, 11], 4)),
        ([1, 2],       ([5], 2)),
        ([1, 2, 7, 8], ([5], 2)),
    ]:
        blocks, m = c.match(prompt)
        assert (blocks, m) == want, f"match({prompt}) = {(blocks, m)}, want {want}"
        alloc.decref(blocks)
    print("  A.4 insert with split + four match cases ✓")

    # A.5 no false hits on unrelated prompt
    blocks, m = c.match([99, 100])
    assert (blocks, m) == ([], 0), f"unrelated match wrong: {(blocks, m)}"
    print("  A.5 no false hits         ✓")

    # A.6 eviction
    # Simulate "all requests finished" by dropping the seed refs we set at A.2/A.4.
    # After this, refcounts represent only the cache's own holds (one ref per block
    # the tree references). Now evict can actually free.
    alloc.decref([5, 7, 11])    # the seed refs from A.2 (5,7) and A.4 (11)
    # Confirm cache still holds these (refcount=1 each, not 0)
    assert all(alloc.refcount[b] == 1 for b in (5, 7, 11)), \
        f"after seed decref, refcounts should be 1: {[alloc.refcount[b] for b in (5,7,11)]}"

    blocks_freed_before = len(alloc.free_blocks)
    try:
        c.evict(1)              # may return int (non-throwing API) or None
    except RuntimeError as e:
        print(f"  A.6 evict raised: {e}")
    blocks_freed_after = len(alloc.free_blocks)
    delta = blocks_freed_after - blocks_freed_before
    assert delta >= 1, (
        f"evict did not free any blocks: delta={delta}; "
        f"refcounts now: 5={alloc.refcount[5]} 7={alloc.refcount[7]} 11={alloc.refcount[11]}"
    )
    print(f"  A.6 evict freed {delta} block(s) ✓")

    print("  Phase A: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — end-to-end with the model
# ─────────────────────────────────────────────────────────────────────────────

def _vram(prefix: str) -> None:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        r = torch.cuda.memory_reserved()  / 2**30
        print(f"  [vram] {prefix}: allocated={a:.2f} GiB  reserved={r:.2f} GiB")


def _free_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PrefillCounter(torch.nn.Module):
    """Forward shim that tallies tokens passed through the model.
    Replaces the model in Scheduler so we can count actual GPU work.
    """
    def __init__(self, model):
        super().__init__()
        self.model   = model
        self.cfg     = model.cfg                 # Scheduler reads .cfg sometimes
        self.total_q = 0
        self.calls   = 0

    def forward(self, input_ids, pool, meta):
        self.calls   += 1
        self.total_q += int(input_ids.numel())
        return self.model(input_ids, pool, meta)

    # let __call__ work without nn.Module overhead surprises
    __call__ = forward

    def reset(self):
        self.total_q = 0
        self.calls   = 0


def run_request(scheduler, prompt_ids: torch.Tensor, rid: int, max_tokens: int) -> list[int]:
    req = Request(
        id=rid,
        prompt_ids=prompt_ids,
        sampling_param=dict(temperature=0.0, top_k=0, top_p=1.0, rep_penalty=1.0),
        max_tokens=max_tokens,
    )
    scheduler.add_request(req)
    out: list[int] = []
    while scheduler.has_unfinished():
        res = scheduler.step()
        if rid in res.new_tokens:
            out.append(res.new_tokens[rid])
        if rid in res.finished:
            break
    return out


def _make_scheduler(model, cache=None):
    cfg   = model.cfg
    alloc = BlockAllocator(NUM_SLOTS)
    pool  = KvPool(cfg.num_hidden_layers, alloc.num_blocks, alloc.block_size,
                   NUM_SLOTS, cfg.num_key_value_heads, cfg.head_dim,
                   dtype=cfg.dtype, device="cuda")
    cache = cache if cache is not None else RadixCache(alloc)
    sched = Scheduler(model, Sampler(), alloc, pool,
                      eos_id=cfg.eos_token_id, cache=cache)
    return sched, cache, alloc, pool


@torch.no_grad()
def phase_b_end_to_end() -> None:
    print("\n=== Phase B: end-to-end with model ===")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    SHARED = "You are a helpful assistant. " * 16        # ~128 tokens
    EXT    = SHARED + " Tell me a joke."

    pid_shared = tok(SHARED, return_tensors="pt").input_ids[0].cuda()
    pid_ext    = tok(EXT,    return_tensors="pt").input_ids[0].cuda()
    print(f"  shared prompt: {pid_shared.numel()} tokens")
    print(f"  ext    prompt: {pid_ext.numel()} tokens")

    print("  loading model...")
    model = load_model(MODEL_DIR); model.eval()
    _vram("after load")
    counted = PrefillCounter(model)

    # ---- B.1: with cache — prime, then ext ----
    print("\n  --- B.1: WITH cache ---")
    sched, cache, alloc_w, pool_w = _make_scheduler(counted)

    counted.reset()
    _ = run_request(sched, pid_shared, rid=0, max_tokens=4)
    prime_q = counted.total_q
    print(f"  prime SHARED prompt: {prime_q} q tokens in {counted.calls} steps")

    counted.reset()
    out_cached = run_request(sched, pid_ext, rid=1, max_tokens=N_NEW)
    q_with_cache = counted.total_q
    print(f"  ext run:  {q_with_cache} q tokens in {counted.calls} steps")
    print(f"  output:   {tok.decode(out_cached)!r}")

    del sched, cache, pool_w, alloc_w
    _free_cuda()

    # ---- B.2: baseline (no cache priming) — fresh allocator, fresh empty cache ----
    print("\n  --- B.2: BASELINE (fresh cache, no priming) ---")
    sched, cache, alloc_b, pool_b = _make_scheduler(counted)

    counted.reset()
    out_baseline = run_request(sched, pid_ext, rid=2, max_tokens=N_NEW)
    q_no_cache = counted.total_q
    print(f"  ext run:  {q_no_cache} q tokens in {counted.calls} steps")
    print(f"  output:   {tok.decode(out_baseline)!r}")

    del sched, cache, pool_b, alloc_b
    _free_cuda()

    # ---- Acceptance ----
    print("\n  --- assertions ---")
    saved = q_no_cache - q_with_cache
    pct   = saved / q_no_cache * 100 if q_no_cache else 0
    assert q_with_cache < q_no_cache, (
        f"Cache did NOT reduce prefill: with_cache={q_with_cache} >= baseline={q_no_cache}"
    )
    print(f"  ✓ cache saved {saved} q tokens ({pct:.1f}% of baseline)")

    assert out_cached == out_baseline, (
        f"Cache changed output!\n  cached:   {out_cached}\n  baseline: {out_baseline}"
    )
    print(f"  ✓ cached output matches baseline bit-for-bit")

    # ---- B.3: unrelated prompt against primed cache → no false hit ----
    print("\n  --- B.3: unrelated prompt should NOT hit ---")
    sched, cache, alloc_u, pool_u = _make_scheduler(counted)
    _ = run_request(sched, pid_shared, rid=10, max_tokens=4)            # prime

    unrelated = tok("Translate to Italian: hello world",
                    return_tensors="pt").input_ids[0].cuda()
    blocks, m = cache.match(unrelated.tolist())
    if m > 0:
        alloc_u.decref(blocks)
    assert m == 0, f"unrelated prompt got a false cache hit: {m} tokens"
    print(f"  ✓ unrelated prompt: matched_len=0")

    del sched, cache, pool_u, alloc_u, model
    _free_cuda()
    _vram("after free")
    print("\n  Phase B: PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        phase_a_unit_tests()
    except AssertionError as e:
        print(f"\n❌ Phase A FAILED: {e}")
        return 1

    if os.environ.get("L8_SKIP_MODEL"):
        print("\n[L8_SKIP_MODEL set; skipping Phase B]")
        print("\n✅ L8 PASS (unit only)")
        return 0

    try:
        phase_b_end_to_end()
    except AssertionError as e:
        print(f"\n❌ Phase B FAILED: {e}")
        return 1

    print("\n✅ L8 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
