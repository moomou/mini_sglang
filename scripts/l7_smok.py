# scripts/l7_smoke.py
import asyncio, json, time, sys
import httpx

URL = "http://localhost:8000/generate"

async def one(prompt, idx):
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", URL, json={"prompt": prompt, "max_tokens": 30}) as r:
            buf = ""
            t0 = time.time(); first_token_t = None
            async for line in r.aiter_lines():
                if not line.startswith("data: "): continue
                if line == "data: [DONE]": break
                first_token_t = first_token_t or (time.time() - t0)
                buf += json.loads(line[6:])["text"]
            print(f"[{idx}] TTFT={first_token_t:.2f}s total={time.time()-t0:.2f}s text={buf!r}")
            return buf

async def main():
    # 3 concurrent requests
    tasks = [one("The capital of France is", 0),
            one("Once upon a time", 1),
            one("Python is", 2)]
    results = await asyncio.gather(*tasks)
    assert all(r for r in results)
    print("✅ L7 PASS")

asyncio.run(main())