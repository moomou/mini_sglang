class BlockAllocator:
    def __init__(self, num_slots, block_size=16):
        assert num_slots % block_size == 0
        self.block_size = block_size
        # slot is resource per "token"
        # block is group of "tokens"
        self.num_blocks = num_slots // block_size
        self.free_blocks = list(range(self.num_blocks))

    def num_free_tokens(self):
        return len(self.free_blocks) * self.block_size

    def alloc_blocks(self, n_blocks):
        if n_blocks > len(self.free_blocks):
            raise RuntimeError(f"OOM: need {n_blocks} blocks, have {len(self.free_blocks)}")

        out = self.free_blocks[-n_blocks:]
        del self.free_blocks[-n_blocks:]
        return out

    def free(self, blocks):
        self.free_blocks.extend(blocks)