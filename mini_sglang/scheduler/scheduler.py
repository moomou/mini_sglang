from mini_sglang.cache.block_alloc import AllocFailure
from mini_sglang.sampler import SamplingParams
import torch
from dataclasses import dataclass
from mini_sglang.cache.request import Request, ForwardMeta, reserve
from collections import deque

# this is GLOBAL # of tokens engine generates
MAX_TOKENS_PER_STEP = 8192
MAX_BATCH_PER_TURN = 32

PREFILL_CHUNK = float('inf')

def next_q_len(req: Request):
    remaining = req.prompt_ids.numel() - req.cur_len
    return min(remaining, PREFILL_CHUNK) if remaining > 0 else 1

@dataclass
class StepResult:
    new_tokens: dict[int, int] # req_id -> tokens
    finished: list[int] # req_ids completed 

class Scheduler:
    def __init__(self, model, sampler, alloc, pool, eos_id, cache=None):
        self.running: deque[Request] = deque()
        self.waiting: deque[Request] = deque()

        self.model = model
        self.sampler = sampler
        self.alloc = alloc
        self.pool=  pool
        self.eos_id = eos_id
        self.cache = cache

        self.cancelled= set()

    def build_params(self, batch: list[tuple[Request, int]]) -> SamplingParams:
        return SamplingParams(
            temperature=torch.tensor([r.sampling_param["temperature"] for r, _ in batch], dtype=torch.float32, device='cuda'),
            top_k=torch.tensor([r.sampling_param["top_k"] for r, _ in batch], dtype=torch.int32, device='cuda'),
            top_p=torch.tensor([r.sampling_param["top_p"] for r, _ in batch], dtype=torch.float32, device='cuda'),
            rep_penalty=torch.tensor([r.sampling_param["rep_penalty"] for r, _ in batch], dtype=torch.float32, device='cuda'),
        )

    def add_request(self, req: Request):
        if self.cache:
            cached_blocks, matched_len = self.cache.match(req.prompt_ids.tolist())
            req.blocks = list(cached_blocks)
            req.cur_len = matched_len
            # also need to udpate slots
            bs = self.alloc.block_size
            for b in cached_blocks:
                req.slot_indices.extend(range(b * bs, b * bs + bs))

        self.waiting.append(req)
    
    def finish(self, rid: int):
        self.cancelled.add(rid)
      
    def has_unfinished(self) -> bool:
        if len(self.waiting) > 0 or len(self.running) > 0:
            return True
        return False

    def _build_input_ids(self, batch: list[tuple[Request, int]]):
        flat = []
        for req, q_len in batch:
            if req.cur_len + q_len <= req.prompt_ids.numel():
                flat.extend(req.prompt_ids[req.cur_len : req.cur_len + q_len].tolist())
            else: 
                flat.append(req.output_ids[-1])

        return torch.tensor(flat, device='cuda', dtype=torch.long)


    def build_prev_ids(self, batch: list[tuple[Request, int]]):
        # calculate max_prev size
        max_prev = max([
            req.prompt_ids.numel() + len(req.output_ids) 
            for req, _ in batch
        ])
        # any int that's not in the vocab penalty math — for L5 with rep_penalty=1.0, anything works
        PAD = 0
        out = torch.full((len(batch), max_prev), PAD, dtype=torch.long, device='cuda')

        for i, (req, _) in enumerate(batch):
            hist = torch.concat([
                req.prompt_ids, 
                torch.tensor(
                    req.output_ids, 
                    dtype=req.prompt_ids.dtype, 
                    device=req.prompt_ids.device,
                )
            ], dim=-1)
            out[i, :hist.numel()] = hist

        return out

    def _step(self, batch: list[tuple[Request, int]]):
        for req, q_len in batch:
            T = req.cur_len + q_len
            reserve(req, self.alloc, T, cache=self.cache)

        meta = self.build_meta(batch)
        params = self.build_params(batch)
        prev_ids = self.build_prev_ids(batch)
        input_ids = self._build_input_ids(batch)

        logits = self.model(input_ids, self.pool, meta)

        # remember, logits is (T_total, vocab)
        last_rows = (meta.cu_seqlens_q[1:] - 1).long()
        # per_seq: (T_seq, vocab_size)
        per_seq = logits[last_rows]
        # next_ids: (T_seq, 1)
        next_ids= self.sampler(
            per_seq.float(),
            params, 
            prev_ids=prev_ids)

        step_result = StepResult({}, finished=[])
        # then update req
        for i, next_id in enumerate(next_ids):
            req , q_len = batch[i]
            next_id_detached= next_id.item()

            req.output_ids.append(next_id_detached)
            req.cur_len += q_len

            step_result.new_tokens[req.id] = next_id_detached

            if next_id_detached == self.eos_id or len(req.output_ids) >= req.max_tokens:
                step_result.finished.append(req.id)

                if self.cache is not None:
                    full_tokens = req.prompt_ids.tolist() + req.output_ids
                    self.cache.insert(full_tokens, req.blocks)

                self._release_req_blocks(req)
            else:
                self.running.append(req)

        return step_result

    def _release_req_blocks(self, req: Request):
        self.cancelled.discard(req.id)
        self.alloc.decref(req.blocks)

        req.blocks.clear()
        req.slot_indices.clear()

    def step(self):
        decode_batch, prefill_batch = self.select()

        batch = prefill_batch + decode_batch

        if not batch:
            return StepResult({}, [])

        return self._step(batch)


    def select(self, target_batch_size=MAX_BATCH_PER_TURN) -> tuple[list[tuple[Request, int]], list[tuple[Request, int]]]:
        target_batch_size = min(target_batch_size, MAX_BATCH_PER_TURN)

        decode_batch = []
        # decode takes exactly 1 step
        while len(self.running) and len(decode_batch) < target_batch_size:
            req = self.running.popleft()
            if req.id in self.cancelled:
                self._release_req_blocks(req)
                continue

            decode_batch.append((req, 1))

        prefill_batch = []
        budget = MAX_TOKENS_PER_STEP - len(decode_batch) # decode is 1 token each
        while self.waiting and budget > 0 and (len(decode_batch) + len(prefill_batch)) < target_batch_size:
            # waiting is prefill
            head = self.waiting[0]
            need = next_q_len(head)

            if head.id in self.cancelled:
                self._release_req_blocks(head)
                self.waiting.popleft()
                continue

            if need > budget:
                # we dont have enough capacity to handle
                break
            if self.alloc.num_free_tokens() < (need + 1): # plus 1 for generated token
                break
            
            prefill_batch.append((self.waiting.popleft(), need))

            # we only process "need" but need plus 1 to save the generated token
            budget -= need

        return decode_batch, prefill_batch

    def build_meta(self, batch: list[tuple[Request, int]]):
        positions = []
        slot_mappings = []
        cu_seqlens_q = [0]
        seq_lens_kv = []

        block_table_rows = []
        max_blocks = max(len(r.blocks) for r, _ in batch)

        for r, q_len in batch:
            new_start = r.cur_len

            for p in range(new_start, new_start + q_len):
                positions.append(p)
                slot_mappings.append(r.slot_indices[p])

            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
            seq_lens_kv.append(r.cur_len + q_len)
            block_table_rows.append(
                r.blocks + [0] * (max_blocks - len(r.blocks))) # pad

        return ForwardMeta(
            positions = torch.tensor(positions, device='cuda', dtype=torch.long),
            slot_mapping = torch.tensor(slot_mappings, device='cuda', dtype=torch.long),
            cu_seqlens_q = torch.tensor(cu_seqlens_q, device='cuda', dtype=torch.int32),
            seq_lens_kv= torch.tensor(seq_lens_kv, device='cuda', dtype=torch.int32),
            block_table= torch.tensor(block_table_rows, device='cuda', dtype=torch.int32),
            block_size = self.alloc.block_size,
        )
