from __future__ import annotations

import hashlib
import math
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple

import torch

PairPriorityFn = Callable[[str, str], float]
LayerKV = Tuple[torch.Tensor, torch.Tensor]
StackedLayers = List[LayerKV]

def canonical_pair_key(doc_id_a: str, doc_id_b: str) -> Tuple[str, str]:
    if doc_id_a < doc_id_b:
        return (doc_id_a, doc_id_b)
    if doc_id_b < doc_id_a:
        return (doc_id_b, doc_id_a)
    return (doc_id_a, doc_id_b)

def pair_hash_key(doc_id_a: str, doc_id_b: str) -> str:
    a, b = canonical_pair_key(doc_id_a, doc_id_b)
    h = hashlib.sha256()
    h.update(a.encode("utf-8"))
    h.update(b"\x00")
    h.update(b.encode("utf-8"))
    return f"pair:{h.hexdigest()}"

class PairKVStore(ABC):
    needs_individuals: bool = False

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
    ) -> Optional[StackedLayers]: ...

    @abstractmethod
    def put(
        self,
        doc_a: str,
        doc_b: str,
        joint_layers: StackedLayers,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> None: ...

    @abstractmethod
    def stats(self) -> dict: ...

    def bytes_used(self) -> int:
        return 0

class FullJointPairStore(PairKVStore):
    def __init__(
        self,
        max_entries: int,
        *,
        bytes_budget: Optional[int] = None,
        store_on_cpu: bool = False,
        cuda_device: int | str | torch.device | None = None,
        fifo: bool = False,
        evict_policy: str = "lru",
        priority_fn: Optional[PairPriorityFn] = None,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.bytes_budget = bytes_budget
        self.store_on_cpu = store_on_cpu
        policy = (evict_policy or "lru").lower()
        if fifo and policy == "lru":
            policy = "fifo"
        if policy not in ("lru", "fifo", "lfu"):
            policy = "lru"
        if priority_fn is None and policy == "lfu":
            priority_fn = lambda a, b: 0.0
        self.evict_policy = policy
        self.priority_fn = priority_fn
        self.fifo = policy == "fifo"
        if cuda_device is None:
            self._cuda_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self._cuda_device = torch.device(cuda_device)
        self._store: "OrderedDict[str, StackedLayers]" = OrderedDict()
        self._pair_names: dict[str, Tuple[str, str]] = {}
        self._lock = threading.RLock()
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
        with self._lock:
            got = self._store.get(key)
            if got is None:
                self.misses += 1
                return None
            if self.evict_policy == "lru":
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

    def put(
        self,
        doc_a: str,
        doc_b: str,
        joint_layers: StackedLayers,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> None:
        del individual_a, individual_b
        key = pair_hash_key(doc_a, doc_b)
        if self.store_on_cpu:
            frozen: StackedLayers = [
                [t[0].detach().cpu().clone(), t[1].detach().cpu().clone()]
                for t in joint_layers
            ]
        else:
            frozen = [[t[0].clone(), t[1].clone()] for t in joint_layers]
        with self._lock:
            if key in self._store:
                if self.evict_policy == "lru":
                    self._store.move_to_end(key)
                return
            self._store[key] = frozen
            self._pair_names[key] = canonical_pair_key(doc_a, doc_b)
            self._evict_to_limits_locked()

    def _evict_to_limits_locked(self) -> None:
        while len(self._store) > self.max_entries:
            self._evict_one()
        if self.bytes_budget is None:
            return
        while len(self._store) > 1 and self._bytes_used_locked() > self.bytes_budget:
            self._evict_one()

    def _evict_one(self) -> None:
        if self.evict_policy == "lfu" and self.priority_fn:
            victim = min(
                self._store.keys(),
                key=lambda k: self.priority_fn(*self._pair_names[k]),  # type: ignore[misc]
            )
            self._store.pop(victim, None)
            self._pair_names.pop(victim, None)
        else:
            victim, _ = self._store.popitem(last=False)
            self._pair_names.pop(victim, None)
        self.evictions += 1

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "entries": len(self._store),
                "capacity": self.max_entries,
                "bytes_budget": self.bytes_budget,
                "kind": "full_joint",
                "fifo": self.fifo,
                "evict_policy": self.evict_policy,
                "bytes_used": self._bytes_used_locked(),
            }

    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used_locked()

    def _bytes_used_locked(self) -> int:
        n = 0
        for entry in self._store.values():
            for k, v in entry:
                n += k.numel() * k.element_size()
                n += v.numel() * v.element_size()
        return n

LayerDelta = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
StackedDeltas = List[LayerDelta]

class SparseDeltaPairStore(PairKVStore):
    needs_individuals: bool = True

    def __init__(
        self,
        max_entries: int,
        *,
        bytes_budget: Optional[int] = None,
        top_k_ratio: float = 0.1,
        store_on_cpu: bool = False,
        cuda_device: int | str | torch.device | None = None,
        evict_policy: str = "lru",
        priority_fn: Optional[PairPriorityFn] = None,
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.bytes_budget = bytes_budget
        r = float(top_k_ratio)
        if r <= 0:
            r = 0.01
        if r > 1:
            r = 1.0
        self.top_k_ratio = r
        self.store_on_cpu = store_on_cpu
        policy = (evict_policy or "lru").lower()
        if policy not in ("lru", "fifo", "lfu"):
            policy = "lru"
        if priority_fn is None and policy == "lfu":
            priority_fn = lambda a, b: 0.0
        self.evict_policy = policy
        self.priority_fn = priority_fn
        if cuda_device is None:
            self._cuda_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self._cuda_device = torch.device(cuda_device)
        self._store: "OrderedDict[str, StackedDeltas]" = OrderedDict()
        self._seq_lens: dict[str, int] = {}
        self._pair_names: dict[str, Tuple[str, str]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.reconstruct_failures = 0

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
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            expected_len = self._seq_lens.get(key)
            if expected_len is None:
                self.misses += 1
                return None
        if individual_a is None or individual_b is None:
            with self._lock:
                self.misses += 1
            return None
        (canon_a, _canon_b) = canonical_pair_key(doc_a, doc_b)
        if canon_a == doc_a:
            base_a, base_b = individual_a, individual_b
        else:
            base_a, base_b = individual_b, individual_a

        joint_layers: StackedLayers = []
        for layer_idx, (indices, dK, dV) in enumerate(entry):
            ka = base_a[layer_idx][0]
            va = base_a[layer_idx][1]
            kb = base_b[layer_idx][0]
            vb = base_b[layer_idx][1]
            device = ka.device
            dtype = ka.dtype
            joint_k = torch.cat((ka, kb), dim=0).clone()
            joint_v = torch.cat((va, vb), dim=0).clone()
            if joint_k.shape[0] != expected_len:
                with self._lock:
                    self.reconstruct_failures += 1
                    self.misses += 1
                return None
            idx = indices.to(device=device, dtype=torch.long, non_blocking=True)
            dk_l = dK.to(device=device, dtype=dtype, non_blocking=True)
            dv_l = dV.to(device=device, dtype=dtype, non_blocking=True)
            joint_k.index_add_(0, idx, dk_l)
            joint_v.index_add_(0, idx, dv_l)
            joint_layers.append([joint_k, joint_v])
        with self._lock:
            if self.evict_policy == "lru" and key in self._store:
                self._store.move_to_end(key)
            self.hits += 1
        return joint_layers

    def put(
        self,
        doc_a: str,
        doc_b: str,
        joint_layers: StackedLayers,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> None:
        if individual_a is None or individual_b is None:
            return
        key = pair_hash_key(doc_a, doc_b)
        with self._lock:
            if key in self._store:
                if self.evict_policy == "lru":
                    self._store.move_to_end(key)
                return

        canon_pair = canonical_pair_key(doc_a, doc_b)
        canon_a = canon_pair[0]
        if canon_a == doc_a:
            base_a, base_b = individual_a, individual_b
        else:
            base_a, base_b = individual_b, individual_a

        if len(joint_layers) != len(base_a) or len(joint_layers) != len(base_b):
            return
        la = base_a[0][0].shape[0]
        lb = base_b[0][0].shape[0]
        total = la + lb
        if joint_layers[0][0].shape[0] != total:
            return

        k_positions = max(1, math.ceil(self.top_k_ratio * total))
        k_positions = min(k_positions, total)

        deltas: StackedDeltas = []
        for layer_idx, (jk, jv) in enumerate(joint_layers):
            ka = base_a[layer_idx][0]
            va = base_a[layer_idx][1]
            kb = base_b[layer_idx][0]
            vb = base_b[layer_idx][1]
            base_k = torch.cat((ka, kb), dim=0)
            base_v = torch.cat((va, vb), dim=0)
            base_k = base_k.to(device=jk.device, dtype=jk.dtype)
            base_v = base_v.to(device=jv.device, dtype=jv.dtype)
            dK = jk - base_k
            dV = jv - base_v
            reduce_dims = tuple(range(1, dK.ndim))
            score = (
                dK.float().pow(2).sum(dim=reduce_dims).sqrt()
                + dV.float().pow(2).sum(dim=reduce_dims).sqrt()
            )
            if k_positions >= total:
                indices = torch.arange(total, device=score.device, dtype=torch.long)
            else:
                indices = torch.topk(score, k=k_positions, largest=True).indices
                indices, _ = torch.sort(indices)
            dK_top = dK.index_select(0, indices).contiguous()
            dV_top = dV.index_select(0, indices).contiguous()
            idx_store = indices.to(dtype=torch.long)
            if self.store_on_cpu:
                deltas.append((
                    idx_store.detach().cpu().clone(),
                    dK_top.detach().cpu().clone(),
                    dV_top.detach().cpu().clone(),
                ))
            else:
                deltas.append((
                    idx_store.detach().clone(),
                    dK_top.detach().clone(),
                    dV_top.detach().clone(),
                ))

        with self._lock:
            if key in self._store:
                if self.evict_policy == "lru":
                    self._store.move_to_end(key)
                return
            self._store[key] = deltas
            self._seq_lens[key] = total
            self._pair_names[key] = canon_pair
            self._evict_to_limits_locked()

    def _evict_to_limits_locked(self) -> None:
        while len(self._store) > self.max_entries:
            self._evict_one()
        if self.bytes_budget is None:
            return
        while len(self._store) > 1 and self._bytes_used_locked() > self.bytes_budget:
            self._evict_one()

    def _evict_one(self) -> None:
        if self.evict_policy == "lfu" and self.priority_fn:
            victim = min(
                self._store.keys(),
                key=lambda k: self.priority_fn(*self._pair_names[k]),  # type: ignore[misc]
            )
            self._store.pop(victim, None)
        else:
            victim, _ = self._store.popitem(last=False)
        self._seq_lens.pop(victim, None)
        self._pair_names.pop(victim, None)
        self.evictions += 1

    def keys(self) -> Iterable[str]:
        return self._store.keys()

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "reconstruct_failures": self.reconstruct_failures,
                "entries": len(self._store),
                "capacity": self.max_entries,
                "bytes_budget": self.bytes_budget,
                "top_k_ratio": self.top_k_ratio,
                "kind": "sparse_delta",
                "evict_policy": self.evict_policy,
                "bytes_used": self._bytes_used_locked(),
            }

    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used_locked()

    def _bytes_used_locked(self) -> int:
        n = 0
        for entry in self._store.values():
            for indices, dK, dV in entry:
                n += indices.numel() * indices.element_size()
                n += dK.numel() * dK.element_size()
                n += dV.numel() * dV.element_size()
        return n
