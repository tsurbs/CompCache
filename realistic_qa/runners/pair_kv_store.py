"""Store for joint KV caches of promoted document pairs.

A pair is keyed by a canonical (sorted) tuple of document identifiers so that
(d_a, d_b) and (d_b, d_a) hash to the same entry. The concrete full-joint
implementation stores the entire concatenation KV; a future sparse-delta
implementation can plug in behind the same interface.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Iterable, List, Optional, Tuple

import torch


LayerKV = Tuple[torch.Tensor, torch.Tensor]
StackedLayers = List[LayerKV]


def canonical_pair_key(doc_id_a: str, doc_id_b: str) -> Tuple[str, str]:
    """Sort two document IDs so pair lookup is order-invariant."""
    if doc_id_a == doc_id_b:
        raise ValueError("pair members must differ")
    return (doc_id_a, doc_id_b) if doc_id_a < doc_id_b else (doc_id_b, doc_id_a)


def pair_hash_key(doc_id_a: str, doc_id_b: str) -> str:
    """Stable string key for dict storage. Includes a prefix to avoid collisions
    with the individual-chunk cache's SHA256 keys."""
    a, b = canonical_pair_key(doc_id_a, doc_id_b)
    h = hashlib.sha256()
    h.update(a.encode("utf-8"))
    h.update(b"\x00")
    h.update(b.encode("utf-8"))
    return f"pair:{h.hexdigest()}"


class PairKVStore(ABC):
    """Abstract interface for pair KV storage.

    Implementations differ on what they store (full joint KV vs sparse delta)
    but all callers should only interact through these methods.
    """

    @abstractmethod
    def contains(self, doc_a: str, doc_b: str) -> bool: ...

    @abstractmethod
    def get(
        self,
        doc_a: str,
        doc_b: str,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> Optional[StackedLayers]:
        """Return the joint per-layer (K, V) for the canonical pair order.

        A delta-store implementation may use the individual caches as base
        tensors onto which it adds the stored delta; the full-joint store
        ignores them.
        """

    @abstractmethod
    def put(self, doc_a: str, doc_b: str, joint_layers: StackedLayers) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...


class FullJointPairStore(PairKVStore):
    """Holds the full concatenated KV for each promoted pair.

    Eviction is LRU (touched on hit); this makes H4's LRU vs LFU comparison
    a single-flag swap later. Storage shape matches FIFOChunkKVCache:
    per-layer list of ``[K, V]`` with shape ``[seq_len, num_kv_heads, head_size]``.

    ``store_on_cpu=True`` stores joint KVs in RAM and stages them to GPU on ``get``.
    """

    def __init__(
        self,
        max_entries: int,
        *,
        store_on_cpu: bool = False,
        cuda_device: int | str | torch.device | None = None,
        fifo: bool = False,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self.store_on_cpu = store_on_cpu
        # ``fifo=True`` evicts in pure insertion order (no LRU-style touch on get
        # or on a put-of-existing-key). Used by the realistic 3-way runner where
        # the pair store is treated as a normal FIFO of joint-KV computations.
        self.fifo = fifo
        if cuda_device is None:
            self._cuda_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self._cuda_device = torch.device(cuda_device)
        self._store: "OrderedDict[str, StackedLayers]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key_tuple: Tuple[str, str]) -> bool:
        return self.contains(key_tuple[0], key_tuple[1])

    def contains(self, doc_a: str, doc_b: str) -> bool:
        return pair_hash_key(doc_a, doc_b) in self._store

    def get(
        self,
        doc_a: str,
        doc_b: str,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> Optional[StackedLayers]:
        key = pair_hash_key(doc_a, doc_b)
        got = self._store.get(key)
        if got is None:
            self.misses += 1
            return None
        if not self.fifo:
            self._store.move_to_end(key)
        self.hits += 1
        return [self._materialize_for_read(t[0], t[1]) for t in got]

    def _materialize_for_read(self, k: torch.Tensor, v: torch.Tensor) -> LayerKV:
        if self.store_on_cpu:
            if torch.cuda.is_available() and self._cuda_device.type == "cuda":
                k = k.to(self._cuda_device, non_blocking=True)
                v = v.to(self._cuda_device, non_blocking=True)
            return [k.clone(), v.clone()]
        return [k.clone(), v.clone()]

    def put(self, doc_a: str, doc_b: str, joint_layers: StackedLayers) -> None:
        key = pair_hash_key(doc_a, doc_b)
        if key in self._store:
            if not self.fifo:
                self._store.move_to_end(key)
            return
        if self.store_on_cpu:
            frozen: StackedLayers = [
                [t[0].detach().cpu().clone(), t[1].detach().cpu().clone()]
                for t in joint_layers
            ]
        else:
            frozen = [[t[0].clone(), t[1].clone()] for t in joint_layers]
        self._store[key] = frozen
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.evictions += 1

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": len(self._store),
            "capacity": self.max_entries,
            "kind": "full_joint",
            "fifo": self.fifo,
        }
