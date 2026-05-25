import torch
from dataclasses import dataclass, field
from mini_sglang.tokenizer import IncrementalDetokenizer

@dataclass
class ForwardMeta:
    # [T] int64, abs pos for rope
    positions: torch.Tensor
    # [T] int64, where to write in KV cache
    slot_mapping: torch.Tensor
    # [num_seqs + 1] int32
    cu_seqlens_q: torch.Tensor
    # [num_seqs] int32
    seq_lens_kv: torch.Tensor
    # alloc block size
    block_size: int
    # [num_seq, max_blocks_per_seq] int32
    block_table: torch.Tensor

    # [T_kv] int64, where to read in KV cache
    # for debugging only: do gathering in CUDA kernels
    kv_indices: torch.Tensor | None = None
    # for SDPA is_causal
    is_prefill: bool = False

@dataclass
class Request:
    id: int
    prompt_ids: torch.Tensor
    sampling_param: dict
    blocks: list[int] = field(default_factory=list)
    slot_indices: list[int] = field(default_factory=list)
    cur_len: int = 0
    output_ids: list[int] = field(default_factory=list)
    max_tokens: int = 20
    done: bool = False
    detok: IncrementalDetokenizer | None = None

def reserve(req: Request, alloc: 'BlockAllocator', target_len: int):
    """Ensure req has enough slots for target_len tokens. Allocates blocks as needed."""
    # convert target-len (per token) into blocks
    target_block = (
        target_len + (alloc.block_size - 1)
    ) // alloc.block_size
    current_block = len(req.blocks)

    if current_block < target_block:
        diff = target_block - current_block
        blocks = alloc.alloc_blocks(diff)
        req.blocks.extend(blocks)

        for b in blocks:
            base = b * alloc.block_size
            req.slot_indices.extend(range(base, base + alloc.block_size))
