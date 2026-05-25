class AllocFailure(Exception):
    ...

class BlockAllocator:
    def __init__(self, num_slots, block_size=16):
        assert num_slots % block_size == 0
        self.block_size = block_size

        # slot is resource per "token"
        # block is group of "tokens"
        self.num_blocks = num_slots // block_size
        self.free_blocks = list(range(self.num_blocks))

        self.refcount = [0] * self.num_blocks

    def num_free_tokens(self):
        return len(self.free_blocks) * self.block_size

    def alloc_blocks(self, n_blocks):
        if n_blocks > len(self.free_blocks):
            raise AllocFailure(f"OOM: need {n_blocks} blocks, have {len(self.free_blocks)}")

        out = self.free_blocks[-n_blocks:]
        del self.free_blocks[-n_blocks:]

        self.incref(out)
        return out

    def incref(self, blocks):
        for b in blocks:
            self.refcount[b] += 1

    def decref(self, blocks):
        freed = 0
        for b in blocks:
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free_blocks.append(b)
                freed += 1
        return freed