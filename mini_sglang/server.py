from typing import Any
import asyncio
import os
import json
import pathlib as pl
from contextlib import asynccontextmanager
from itertools import count

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mini_sglang.config import ModelConfig
from mini_sglang.weights import load_model
from mini_sglang.sampler import Sampler, SamplingParams
from mini_sglang.scheduler.scheduler import Scheduler
from mini_sglang.cache.kv_pool import KvPool
from mini_sglang.cache.block_alloc import BlockAllocator
from mini_sglang.cache.request import Request
from mini_sglang.tokenizer import load_tokenizer, IncrementalDetokenizer
from mini_sglang.engine import Engine, GenRequest

MODEL_DIR = pl.Path(os.environ.get("MODEL_DIR", "/media/2nvme/llm/Qwen3-8B"))
NUM_SLOTS = 8192

class GenerateBody(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    rep_penalty: float = 1.0
    stream: bool = True

state: dict[str, Any] = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model = load_model(MODEL_DIR)
    model.eval()
    cfg: Any = model.cfg

    tok = load_tokenizer(MODEL_DIR)
    alloc = BlockAllocator(NUM_SLOTS)
    pool = KvPool(
        cfg.num_hidden_layers, 
        alloc.num_blocks,
        alloc.block_size,
        NUM_SLOTS, cfg.num_key_value_heads, cfg.head_dim, dtype=cfg.dtype, device='cuda')

    sched = Scheduler(
        model,
        Sampler(),
        alloc,
        pool, 
        eos_id=cfg.eos_token_id)
    engine = Engine(sched, tok)
    engine.start()

    state.update(engine=engine, tokenizer=tok, next_id=count())

    print("ready")
    yield
    print("engine shutting down")
    engine.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(body: GenerateBody):
    engine = state["engine"]
    tok = state["tokenizer"]
    rid = next(state["next_id"])

    prompt_ids = tok(body.prompt, return_tensors="pt").input_ids[0].cuda()
    detok = IncrementalDetokenizer(tok, prompt_ids)

    req = Request(
        id=rid,
        prompt_ids=prompt_ids,
        sampling_param=dict(temperature=body.temperature, top_k=body.top_k,
                               top_p=body.top_p, rep_penalty=body.rep_penalty),
        max_tokens=body.max_tokens,
        detok=detok,
    )

    out_q = asyncio.Queue()
    gen = GenRequest(req=req, out_q=out_q, loop=asyncio.get_running_loop())
    engine.submit(gen)

    async def event_stream():
        try:
            while True:
                piece = await out_q.get()
                if piece is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps({'text': piece})}\n\n"
        except asyncio.CancelledError:
            # client disconnected
            engine.cancel(rid)
            raise

    if body.stream:
        return StreamingResponse(
            event_stream(), media_type="text/event-stream")

    chunks = [] 
    while True:
        p = await out_q.get()
        if p is None:
            break
        chunks.append(p)

    return {'text': "".join(chunks)}