import queue
import asyncio
import threading
from mini_sglang.cache.request import Request
from dataclasses import dataclass

@dataclass
class GenRequest:
    req: Request
    out_q: asyncio.Queue
    loop: asyncio.AbstractEventLoop

class Engine:
    def __init__(self, scheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        self.alive = threading.Event()
        self.alive.set()

        self.thread = threading.Thread(target=self._run, daemon=True)

        self.pending_greq = queue.Queue()
        self.inflight: dict[str, GenRequest] = dict()
        self.cancelled = set()

    def cancel(self, rid):
        self.cancelled.add(rid)

        gen = self.inflight.pop(rid, None)
        if gen:
            self._send(gen, None)

    def start(self):
        self.thread.start()

    def stop(self):
        self.alive.clear()
        self.thread.join(timeout=5)

    def submit(self, gen: GenRequest):
        self.pending_greq.put_nowait(gen)

    def _run(self):
        while self.alive.is_set():
            if self.scheduler.has_unfinished():
                while True:
                    try:
                        gen= self.pending_greq.get_nowait()
                        self.inflight[gen.req.id] = gen
                        self.scheduler.add_request(gen.req)
                    except queue.Empty:
                        break
            else:
                try:
                    gen= self.pending_greq.get(timeout=1.0)
                    self.inflight[gen.req.id] = gen
                    self.scheduler.add_request(gen.req)
                except queue.Empty:
                    if not self.scheduler.has_unfinished():
                        continue

            res = self.scheduler.step()
            for rid, tok in res.new_tokens.items():
                if rid in self.cancelled:
                    self.scheduler.finish(rid)
                    self.inflight.pop(rid, None)
                    self.cancelled.discard(rid)
                    continue

                # cancel may have removed the inflight
                gen = self.inflight.get(rid, None)
                if gen is None:
                    continue

                piece = gen.req.detok.push(tok)
                if piece:
                    self._send(gen, piece)

            for rid in res.finished:
                gen = self.inflight.pop(rid, None)
                if gen is None:
                    continue

                tail = gen.req.detok.flush()
                if tail:
                    self._send(gen, tail)

                # finish the request
                self._send(gen, None)

    def _send(self, gen: GenRequest, item):
        gen.loop.call_soon_threadsafe(
            gen.out_q.put_nowait, item)