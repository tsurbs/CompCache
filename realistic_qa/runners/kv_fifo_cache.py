"""FIFO cache for per-chunk KV tensors (CacheBlend independent chunk collection)."""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import torch


LayerKV = Tuple[torch.Tensor, torch.Tensor]
StackedLayers = List[LayerKV]


class FIFOChunkKVCache:
    """Maps chunk keys to cloned per-layer [K,V] lists; evicts first-inserted entries."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._store: dict[str, StackedLayers] = {}
        self._fifo: deque[str] = deque()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    def _evict_until_ok(self) -> None:
        while len(self._store) > self.max_entries and self._fifo:
            k = self._fifo.popleft()
            if k in self._store:
                del self._store[k]
                self.evictions += 1

    def get(self, key: str) -> StackedLayers | None:
        got = self._store.get(key)
        if got is not None:
            self.hits += 1
            return [[t[0].clone(), t[1].clone()] for t in got]
        self.misses += 1
        return None

    def put(self, key: str, layers: StackedLayers) -> None:
        if key in self._store:
            return
        frozen: StackedLayers = [[t[0].clone(), t[1].clone()] for t in layers]
        self._store[key] = frozen
        self._fifo.append(key)
        self._evict_until_ok()

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": len(self._store),
        }
