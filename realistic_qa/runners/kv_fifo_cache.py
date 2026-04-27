from __future__ import annotations

from collections import deque
from typing import List, Tuple

import torch

LayerKV = Tuple[torch.Tensor, torch.Tensor]
StackedLayers = List[LayerKV]

class FIFOChunkKVCache:
    def __init__(
        self,
        max_entries: int,
        *,
        store_on_cpu: bool = False,
        cuda_device: int | str | torch.device | None = None,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.store_on_cpu = store_on_cpu
        if cuda_device is None:
            self._cuda_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self._cuda_device = torch.device(cuda_device)
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

    def _materialize_for_read(self, k: torch.Tensor, v: torch.Tensor) -> LayerKV:
        if self.store_on_cpu:
            if torch.cuda.is_available() and self._cuda_device.type == "cuda":
                k = k.to(self._cuda_device, non_blocking=True)
                v = v.to(self._cuda_device, non_blocking=True)
            return [k.clone(), v.clone()]
        return [k.clone(), v.clone()]

    def get(self, key: str) -> StackedLayers | None:
        got = self._store.get(key)
        if got is not None:
            self.hits += 1
            return [self._materialize_for_read(t[0], t[1]) for t in got]
        self.misses += 1
        return None

    def put(self, key: str, layers: StackedLayers) -> None:
        if key in self._store:
            return
        if self.store_on_cpu:
            frozen = [
                [t[0].detach().cpu().clone(), t[1].detach().cpu().clone()]
                for t in layers
            ]
        else:
            frozen = [[t[0].clone(), t[1].clone()] for t in layers]
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
