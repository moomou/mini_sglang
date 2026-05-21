import torch

class KvPool:
    def __init__(self, 
            num_layer, 
            num_blocks,
            block_size,
            num_slots, 
            num_kv_heads, 
            head_dim, 
            dtype: torch.dtype, 
            device="cuda"):
        self.num_layer = num_layer
        # slot is per token
        self.num_slots= num_slots
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.K = [
            torch.empty(
                num_blocks,
                block_size, 
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ) for _ in range(self.num_layer)
        ]
        self.V = [
            torch.empty(
                num_blocks,
                block_size, 
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ) for _ in range(self.num_layer)
        ]

    def set_kv(self, 
            layer_id: int, 
            slot_mapping: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor):
        K = self.K[layer_id].view(-1, self.num_kv_heads, self.head_dim)
        V = self.V[layer_id].view(-1, self.num_kv_heads, self.head_dim)

        K[slot_mapping] = k
        V[slot_mapping] = v

    def get_kv(self, layer_id: int, slot_mapping: torch.Tensor):
        # NOTE: only for debugging
        k = self.K[layer_id][slot_mapping]
        v = self.V[layer_id][slot_mapping]
        return k, v

