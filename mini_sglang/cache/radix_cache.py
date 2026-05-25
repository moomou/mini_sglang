import time
from dataclasses import dataclass
from collections import defaultdict

from .block_alloc import BlockAllocator

@dataclass
class Node:
    tokens: list[int]
    blocks: list[int]
    children: dict[int, "Node"]
    parent: "Node | None"
    last_access_time: float = 0.0
    level: int = 0

class RadixCache:
    def __init__(self, alloc):
        self.alloc = alloc

        self.node_by_level: dict[int, list[Node]] = defaultdict(list)
        self.root = Node(
            tokens=[], 
            blocks=[], 
            children={}, 
            parent=None, 
            last_access_time=0.0,
            level=0)

    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        node = self.root
        out_blocks = []
        matched = 0

        while matched < len(token_ids):
            first = token_ids[matched]
            child = node.children.get(first)
            if child is None:
                break

            cp = self._common_prefix_len(child.tokens, token_ids[matched:])
            child.last_access_time = time.time()

            if cp == len(child.tokens):
                out_blocks.extend(child.blocks)
                matched += cp
                node = child
            else:
                blk = cp // self.alloc.block_size
                out_blocks.extend(child.blocks[:blk])
                matched += cp
                break

        if out_blocks:
            self.alloc.incref(out_blocks)

        return out_blocks, matched

    def _common_prefix_len(self, a: list[int], b: list[int]):
        i = 0
        j = 0

        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                break

        bs = self.alloc.block_size
        return bs * (i // bs)

    def insert(self, token_ids, blocks):
        ''' walk the tree and split and insert blocks as necessary '''
        node = self.root
        matched = 0

        while matched < len(token_ids):
            first = token_ids[matched]
            child = node.children.get(first)
            if not child:
                break

            cp= self._common_prefix_len(child.tokens, token_ids[matched:])
            if cp == len(child.tokens):
                matched += cp
                node = child
            else:
                # partial
                matched += cp
                if cp < len(child.tokens):
                    node = self._split(child, cp)

                break

        if token_ids[matched:]:
            blk_idx = matched // self.alloc.block_size

            # add new child
            child = Node(
                tokens=token_ids[matched:], 
                blocks=blocks[blk_idx:],
                children={}, 
                parent=node, 
                last_access_time=time.time(),
                level=node.level + 1)

            self.node_by_level[child.level].append(child)

            node.children[child.tokens[0]] = child
            self.alloc.incref(child.blocks)

    def evict(self, n_blocks):
        # collect the tree in reverse order 
        # collect all leafs, sort by last access time and recycle
        # continue level by level until we recycled n_blocks
        levels = sorted(self.node_by_level.keys())
        target = n_blocks
        while n_blocks > 0 and levels:
            level = levels.pop()
            nodes = sorted(
                self.node_by_level[level], key=lambda node: node.last_access_time)

            if not nodes:
                continue

            for i, n in enumerate(nodes):
                # recycle the blocks
                freed = self.alloc.decref(n.blocks)
                # delete reference to node
                first_token = n.tokens[0]
                if first_token in n.parent.children:
                    del n.parent.children[first_token]

                n_blocks -= freed
                if n_blocks <= 0:
                    break

            self.node_by_level[level] = nodes[i+1:]
        return target - n_blocks


    def _split(self, node, k):
        # k must be blocked aligned
        assert k != 0
        assert k % self.alloc.block_size == 0

        blk_idx = k // self.alloc.block_size

        head_token_ids, tail_token_ids = node.tokens[:k], node.tokens[k:]
        head_blocks, tail_blocks = node.blocks[:blk_idx], node.blocks[blk_idx:]

        split_node = Node(
            tokens=tail_token_ids,
            blocks=tail_blocks,
            children=node.children,
            parent=node,
            last_access_time=time.time(),
            level=node.level + 1
        )

        for child in node.children.values():
            child.parent = split_node

        self.node_by_level[split_node.level].append(split_node)

        node.tokens = head_token_ids
        node.blocks = head_blocks
        node.children=  {split_node.tokens[0]: split_node}

        return node