# mini-sglang

An MVP rebuild of [sglang](https://github.com/sgl-project/sglang) for learning purposes.
Single GPU, bf16, Qwen3-only. Intentionally minimal so the architecture is legible.

## Target hardware

This template is configured for a single NVIDIA GPU. PyTorch is pinned to the
**CUDA 12.8** wheel index because the development machine has an **RTX 5090
(Blackwell, sm_120)**. Older CUDA wheels do not support sm_120.

If you are on Hopper / Ada / Ampere you can also use the cu128 wheels, or relax
`tool.uv.sources.torch` in `pyproject.toml` to the default cu124 index.

## Setup

```bash
# from this directory
uv venv --python 3.12
source .venv/bin/activate
uv sync                       # installs torch + everything except flash-attn
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `2.7.x+cu128 True NVIDIA GeForce RTX 5090` (or your GPU).

### Lesson 3 only — build flash-attn

```bash
uv pip install -e ".[flash]" --no-build-isolation
# ~10 min compile. Needs nvcc available; if missing, install CUDA toolkit 12.8.
```

flash-attn is **not needed for Lessons 1–2**. The L1 baseline uses `torch.nn.functional.scaled_dot_product_attention`, which is correct and fast enough on a 5090.

## Layout

```
mini_sglang/
├── pyproject.toml
├── README.md
├── docs/                          # static HTML lesson pages (L0–L8)
├── mini_sglang/
│   ├── __init__.py
│   ├── config.py                  # ModelConfig (L1)
│   ├── weights.py                 # safetensors loader (L1)
│   ├── sampler.py                 # greedy / temp / top-p / rep penalty (L4)
│   ├── tokenizer.py               # incremental UTF-8 detok (L6)
│   ├── engine.py                  # background loop bridging scheduler ↔ HTTP (L7)
│   ├── server.py                  # FastAPI /generate (L7)
│   ├── model/
│   │   ├── layers.py              # RMSNorm, SwiGLU MLP, RoPE (L1)
│   │   └── qwen3.py               # Qwen3ForCausalLM forward (L1+, paged from L3)
│   ├── cache/
│   │   ├── block_alloc.py         # block allocator (L2)
│   │   ├── kv_pool.py             # paged KV pool (L2)
│   │   ├── request.py             # Request + ForwardMeta (L2/L5)
│   │   └── radix_cache.py         # prefix radix cache (L8)
│   └── scheduler/
│       └── scheduler.py           # FCFS continuous batcher (L5)
└── scripts/
    ├── l1_smoke.py                # greedy 20 tokens, asserts == HF (L1)
    ├── l2_smoke.py                # same parity check on paged KV (L2)
    ├── l3_smoke.py                # paged attention parity (L3)
    ├── l4_smoke.py                # sampler (L4)
    ├── l5_smoke.py                # scheduler / continuous batching (L5)
    ├── l6_smoke.py                # incremental detok across CJK + emoji (L6)
    ├── l7_smok.py                 # FastAPI server, 3 concurrent streams (L7)
    └── l8_smoke.py                # radix prefix cache unit + e2e (L8)
```

## Smoke tests

The model path defaults to `/media/2nvme/llm/Qwen3-8B`. Override with `MODEL_DIR=...`.

The latest smoke test (currently `l8_smoke.py`) is the one that exercises the
current code end-to-end:

```bash
# after `uv sync`  (and `uv pip install -e ".[flash]"` for L3+)
python -m scripts.l8_smoke        # full radix-cache test, requires GPU + model
L8_SKIP_MODEL=1 python -m scripts.l8_smoke   # Phase A unit tests only, no GPU needed
```

> ⚠️ **Caution: earlier smoke tests are frozen against older APIs and will
> not all run against `main`.** Each lesson moves the internal API forward
> (e.g. the model went from `model(ids, positions, kv_caches, ...)` in L1 to
> `model(ids, pool, ForwardMeta)` from L2 onward; `ForwardMeta` then gained
> `cu_seqlens_q` / `seq_lens_kv` / `block_table` at L3, etc.). The scripts
> under `scripts/` are kept as historical checkpoints of what each lesson
> shipped, not as a regression suite. **Only the latest `lN_smoke.py` is
> guaranteed to run against the current code.** Older ones may fail to
> import or raise `TypeError` on the model call.
