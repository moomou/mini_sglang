from idna import decode
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
    def __init__(self, model, sampler, alloc, pool, eos_id):
        self.running: deque[Request] = deque()
        self.waiting: deque[Request] = deque()

        self.model = model
        self.sampler = sampler
        self.alloc = alloc
        self.pool=  pool
        self.eos_id = eos_id

    def build_params(self, batch: list[tuple[Request, int]]) -> SamplingParams:
        return SamplingParams(
            temperature=torch.tensor([r.sampling_param["temperature"] for r, _ in batch], dtype=torch.float32, device='cuda'),
            top_k=torch.tensor([r.sampling_param["top_k"] for r, _ in batch], dtype=torch.int32, device='cuda'),
            top_p=torch.tensor([r.sampling_param["top_p"] for r, _ in batch], dtype=torch.float32, device='cuda'),
            rep_penalty=torch.tensor([r.sampling_param["rep_penalty"] for r, _ in batch], dtype=torch.float32, device='cuda'),
        )

    def add_request(self, req: Request):
        self.waiting.append(req)
      
    def has_unfinished(self) -> bool:
        if len(self.waiting) > 0 or len(self.running) > 0:
            return True
        return False

    def _build_input_ids(self, reqs: list[tuple[Request, int]]):
        ...

    def _build_slots(self, reqs: list[tuple[Request, int]]):
        ...
        slot = torch.tensor(req.slot_indices[:T], device="cuda", dtype=torch.long)

    def build_prev_ids(self, reqs: list[tuple[Request, int]]):
        # torch.tensor([req.prompt_ids], device='cuda')
        ...

    def _step(self, batch: list[tuple[Request, int]]):
        for req, _ in batch:
            T = req.cur_len
            reserve(req, self.alloc, T)

        slots = self._build_slots(batch)
        meta = self.build_meta(batch)
        params = self.build_params(batch)
        prev_ids = self.build_prev_ids(batch)

        logits = self.model(batch, self.pool, meta)

        # remember, logits is (T_total, vocab)
        last_rows = (meta.cu_seqlens_q[1:] - 1).long()
        # per_seq: (T_seq, vocab_size)
        per_seq = logits[last_rows]
        # next_ids: (T_seq, 1)
        next_ids= self.sampler(
            per_seq.float(),
            params, 
            prev_ids=prev_ids)

        # TODO: build a mask to pick out the next_ids for each of the sequence in the batch
        # then update req
        for i, next_id in enumerate(next_ids):
            req , q_len = batch[i]
            req.output_ids.append(next_id.item())
            req.cur_len += q_len

    def step(self):
        decode_batch, prefill_batch = self.select()

        batch = prefill_batch + decode_batch
        self._step(batch)

        # TODO: return StepResult


    def select(self, target_batch_size=MAX_BATCH_PER_TURN) -> tuple[list[tuple[Request, int]], list[tuple[Request, int]]]:
        target_batch_size = min(target_batch_size, MAX_BATCH_PER_TURN)
        decode_batch = []
        while len(self.running) and target_batch_size:
            decode_batch.append((self.running.popleft(), 1))

        prefill_batch = []
        budget = MAX_TOKENS_PER_STEP - len(decode_batch) # decode is 1 token each
        while self.waiting and budget > 0:
            # waiting is prefill
            head = self.waiting[0]
            need = next_q_len(head)

            if need > budget:
                break
            if self.alloc.num_free_tokens() < need + 1: # plus 1 for generated token
                break
            
            prefill_batch.append((self.waiting.popleft(), need))
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
            is_prefill = any(q_len > 1 for _, q_len in batch), # TODO: remove
        )
