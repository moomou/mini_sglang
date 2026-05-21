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
├── mini_sglang/
│   ├── __init__.py
│   ├── config.py              # ModelConfig, ServerArgs (L1)
│   ├── weights.py             # safetensors loader (L1)
│   ├── model/
│   │   ├── layers.py          # RMSNorm, SwiGLU MLP, RoPE (L1)
│   │   └── qwen3.py           # Qwen3ForCausalLM eager forward (L1)
│   ├── cache/
│   │   └── kv_pool.py         # block allocator + page tables (L2)
│   ├── scheduler/
│   │   ├── request.py         # Req state machine (L5)
│   │   ├── batch.py           # ScheduleBatch / ForwardBatch (L5)
│   │   └── scheduler.py       # FCFS continuous batcher (L5)
│   ├── runner/
│   │   ├── model_runner.py    # forward() driver (L3)
│   │   └── sampler.py         # greedy / temp / top-p (L4)
│   └── io/
│       ├── tokenizer.py       # incremental UTF-8 detok (L6)
│       └── server.py          # FastAPI /generate (L7)
├── scripts/
│   └── l1_smoke.py            # greedy 20 tokens, asserts == HF
└── tests/
```

## Lesson 1 quickstart

```bash
# after `uv sync`
python -m scripts.l1_smoke    # should print 20 tokens and "✅ L1 PASS"
```

The model path is hardcoded to `/media/2nvme/llm/Qwen3-8B`. Override with `MODEL_DIR=...`.
