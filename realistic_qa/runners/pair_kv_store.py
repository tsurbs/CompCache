"""Store for joint KV caches of promoted document pairs.

A pair is keyed by a canonical (sorted) tuple of document identifiers so that
(d_a, d_b) and (d_b, d_a) hash to the same entry. Two concrete implementations
ship in this module:

* :class:`FullJointPairStore` — the §3.1 store: keeps the whole
  concatenation KV tensor per pair. Fast path, high memory.
* :class:`SparseDeltaPairStore` — the §3.2 store: keeps only
  ``Δ = joint_KV - cat(individual_KV(a), individual_KV(b))`` sparsified
  to the top-K positions (by combined ‖ΔK‖+‖ΔV‖ per layer). The caller
  passes the current individual caches on ``get``/``put`` so the store
  can reconstruct the joint on demand.

Both implementations share the same :class:`PairKVStore` interface so the
composition-cache assembler treats them interchangeably.
"""
from __future__ import annotations

import hashlib
import math
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Callable, Iterable, List, Optional, Tuple

import torch


# LFU priority callback: given the canonical (doc_a, doc_b) string pair,
# return a scalar "importance" score. On eviction the store keeps the
# entries with the largest scores (i.e. pops the smallest). Typical
# implementations consult a CoRetrievalLogger or a retrieval-frequency
# table maintained by the caller.
PairPriorityFn = Callable[[str, str], float]


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

    Subclasses may set ``needs_individuals = True`` to signal that the
    composition cache must pass the individual chunk KVs on both
    :meth:`get` and :meth:`put`; the orchestrator inspects this flag and
    fetches the individuals from the FIFO (or runs a forward) before
    calling. Full-joint stores leave it ``False`` and the individuals are
    ignored.
    """

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
    ) -> Optional[StackedLayers]:
        """Return the joint per-layer (K, V) for the canonical pair order.

        A delta-store implementation uses the individual caches as base
        tensors onto which it adds the stored delta; the full-joint store
        ignores them.
        """

    @abstractmethod
    def put(
        self,
        doc_a: str,
        doc_b: str,
        joint_layers: StackedLayers,
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> None:
        """Store ``joint_layers`` for the pair. Delta stores additionally
        require the individual KVs so they can persist the delta, not the
        whole joint."""

    @abstractmethod
    def stats(self) -> dict: ...

    def bytes_used(self) -> int:
        """Total element bytes held across all cached entries.

        Default implementation returns 0; concrete stores override to walk
        their internal tensors. Used by memory-savings benchmarks to track
        the cache footprint over time without relying on RSS.
        """
        return 0


class FullJointPairStore(PairKVStore):
    """Holds the full concatenated KV for each promoted pair.

    Eviction is LRU (touched on hit); this makes H4's LRU vs LFU comparison
    a single-flag swap later. Storage shape matches FIFOChunkKVCache:
    per-layer list of ``[K, V]`` with shape ``[seq_len, num_kv_heads, head_size]``.

    ``store_on_cpu=True`` stores joint KVs in RAM and stages them to GPU on ``get``.

    ``evict_policy`` selects the eviction rule when ``len > max_entries``:

    - ``"lru"`` (default): drop the least-recently-used entry.  ``get`` /
      ``put`` on an existing key re-inserts at the most-recent end.
    - ``"fifo"``: drop the first-inserted entry; reads do not refresh order.
      (``fifo=True`` is kept as a backwards-compatible alias.)
    - ``"lfu"``: drop the entry with the minimum priority as reported by
      ``priority_fn(doc_a, doc_b)``.  Caller owns the frequency source
      (e.g. ``CoRetrievalLogger.count``); ties break by insertion order
      (older loses).
    """

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
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if bytes_budget is not None and bytes_budget < 1:
            raise ValueError("bytes_budget must be >= 1 when set")
        self.max_entries = max_entries
        # Optional soft bytes cap.  When set the store evicts entries (per
        # ``evict_policy``) until ``bytes_used <= bytes_budget`` after every
        # admission.  This is how the memory-savings benchmark gets a fair
        # comparison: Full-Joint and Sparse-Δ share the same byte budget,
        # so the smaller representation mechanically fits more pairs.
        self.bytes_budget = bytes_budget
        self.store_on_cpu = store_on_cpu
        policy = (evict_policy or "lru").lower()
        if fifo and policy == "lru":
            policy = "fifo"
        if policy not in ("lru", "fifo", "lfu"):
            raise ValueError(f"evict_policy must be 'lru', 'fifo', or 'lfu'; got {evict_policy!r}")
        if policy == "lfu" and priority_fn is None:
            raise ValueError("evict_policy='lfu' requires priority_fn")
        self.evict_policy = policy
        self.priority_fn = priority_fn
        self.fifo = policy == "fifo"  # keep the public attribute for back-compat
        if cuda_device is None:
            self._cuda_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self._cuda_device = torch.device(cuda_device)
        self._store: "OrderedDict[str, StackedLayers]" = OrderedDict()
        # Map hash-key → canonical (doc_a, doc_b) so LFU eviction can
        # query ``priority_fn`` using the original document identifiers.
        self._pair_names: dict[str, Tuple[str, str]] = {}
        # PromotionWorker writes from a background thread while the main
        # eval loop reads/touches the store; an ``RLock`` guards the
        # OrderedDict so we never call ``move_to_end`` on a key that was
        # just evicted by a concurrent ``put``.  Reentrant so ``put`` can
        # call ``_evict_one`` without releasing.
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
            # Only LRU refreshes recency on reads; FIFO and LFU leave
            # insertion order untouched so reads don't preempt
            # frequency-based eviction.  ``key`` is guaranteed present
            # under the lock.
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
        # Full-joint store ignores individuals; they are accepted so the
        # orchestrator can call put() uniformly across store kinds.
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
        """Evict until both the entry cap and the byte budget are satisfied.

        Must be called under ``self._lock``.  Entry cap is enforced first
        (cheap; no scan), then byte budget (recomputes ``bytes_used`` each
        round because the stored tensors can be highly variable in size).
        If a single entry exceeds the budget on its own we leave it in and
        stop evicting — the alternative is either silently dropping the
        just-inserted entry (surprising) or recursing forever (bug).
        """
        while len(self._store) > self.max_entries:
            self._evict_one()
        if self.bytes_budget is None:
            return
        while (
            len(self._store) > 1
            and self._bytes_used_locked() > self.bytes_budget
        ):
            self._evict_one()

    def _evict_one(self) -> None:
        if self.evict_policy == "lfu":
            assert self.priority_fn is not None
            # Min-priority wins; ties break by insertion order via the
            # ``OrderedDict`` iteration order (older keys come first).
            victim = min(
                self._store.keys(),
                key=lambda k: self.priority_fn(*self._pair_names[k]),  # type: ignore[misc]
            )
            self._store.pop(victim, None)
            self._pair_names.pop(victim, None)
        else:  # "lru" and "fifo" both evict from the head
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
        total = 0
        for entry in self._store.values():
            for k, v in entry:
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total


# One layer's sparse delta entry:
#   indices : LongTensor[K]             — positions into [0, L_a + L_b)
#   dK      : Tensor[K, n_kv_heads, hd] — joint_K[pos] - base_K[pos]
#   dV      : Tensor[K, n_kv_heads, hd] — joint_V[pos] - base_V[pos]
LayerDelta = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
StackedDeltas = List[LayerDelta]


class SparseDeltaPairStore(PairKVStore):
    """Stores ``Δ = joint_KV − cat(individual_KV_a, individual_KV_b)`` sparsified
    to the top-K positions per layer (Proposal §3.2).

    For each layer we keep only the rows (seq-positions) where the combined
    score ``‖ΔK[pos]‖_2 + ‖ΔV[pos]‖_2`` is largest. ``K = max(1, ceil(
    top_k_ratio × (L_a + L_b)))``. On :meth:`get` the store reconstructs
    the joint tensor as ``base = cat(ind_a, ind_b); base.index_add_(0,
    indices, delta)`` — so the caller must pass the current individual
    KVs (pulled from the FIFO or just collected).

    ``needs_individuals = True`` tells the composition orchestrator it
    must supply ``individual_a`` / ``individual_b`` on both :meth:`get`
    and :meth:`put`. If either is missing the store raises on ``put``
    (callers should check with :attr:`needs_individuals` or the FIFO
    first) and returns ``None`` on ``get`` (treated as a miss).

    Eviction is LRU (touched on hit) by default, matching
    :class:`FullJointPairStore`.  Set ``evict_policy="lfu"`` + a
    ``priority_fn`` to drop the least-popular pair when capacity is hit
    (the budget-aware realistic runner uses this to keep high-value pairs
    longer under a fixed bytes budget).
    """

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
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if bytes_budget is not None and bytes_budget < 1:
            raise ValueError("bytes_budget must be >= 1 when set")
        if not 0.0 < top_k_ratio <= 1.0:
            raise ValueError("top_k_ratio must be in (0, 1]")
        policy = (evict_policy or "lru").lower()
        if policy not in ("lru", "fifo", "lfu"):
            raise ValueError(f"evict_policy must be 'lru', 'fifo', or 'lfu'; got {evict_policy!r}")
        if policy == "lfu" and priority_fn is None:
            raise ValueError("evict_policy='lfu' requires priority_fn")
        self.max_entries = max_entries
        self.bytes_budget = bytes_budget
        self.top_k_ratio = float(top_k_ratio)
        self.store_on_cpu = store_on_cpu
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
        self._seq_lens: "dict[str, int]" = {}  # key → L_a + L_b
        self._pair_names: dict[str, Tuple[str, str]] = {}
        # Same rationale as :class:`FullJointPairStore`: the
        # PromotionWorker writes from a background thread.
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.reconstruct_failures = 0  # shape mismatch on get

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key_tuple: Tuple[str, str]) -> bool:
        return self.contains(key_tuple[0], key_tuple[1])

    def contains(self, doc_a: str, doc_b: str) -> bool:
        return pair_hash_key(doc_a, doc_b) in self._store

    # ---- get -------------------------------------------------------------

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
            # Cannot reconstruct without base tensors. Treat as a miss so
            # the orchestrator falls back to individual assembly.
            with self._lock:
                self.misses += 1
            return None
        # Canonicalize the caller's individuals to the stored order.
        (canon_a, _canon_b) = canonical_pair_key(doc_a, doc_b)
        if canon_a == doc_a:
            base_a, base_b = individual_a, individual_b
        else:
            base_a, base_b = individual_b, individual_a

        joint_layers: StackedLayers = []
        try:
            for layer_idx, (indices, dK, dV) in enumerate(entry):
                ka = base_a[layer_idx][0]
                va = base_a[layer_idx][1]
                kb = base_b[layer_idx][0]
                vb = base_b[layer_idx][1]
                # The individuals may live on any device/dtype; match the
                # delta against them.
                device = ka.device
                dtype = ka.dtype
                joint_k = torch.cat((ka, kb), dim=0).clone()
                joint_v = torch.cat((va, vb), dim=0).clone()
                if joint_k.shape[0] != expected_len:
                    raise RuntimeError(
                        f"individual lengths don't match stored pair "
                        f"({joint_k.shape[0]} vs {expected_len})"
                    )
                idx = indices.to(device=device, dtype=torch.long, non_blocking=True)
                dk_l = dK.to(device=device, dtype=dtype, non_blocking=True)
                dv_l = dV.to(device=device, dtype=dtype, non_blocking=True)
                joint_k.index_add_(0, idx, dk_l)
                joint_v.index_add_(0, idx, dv_l)
                joint_layers.append([joint_k, joint_v])
        except Exception:
            with self._lock:
                self.reconstruct_failures += 1
                self.misses += 1
            return None
        with self._lock:
            # ``key`` may have been evicted between the read above and
            # here; tolerate that — the reconstructed joint is still valid
            # to return to this caller.
            if self.evict_policy == "lru" and key in self._store:
                self._store.move_to_end(key)
            self.hits += 1
        return joint_layers

    # ---- put -------------------------------------------------------------

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
            raise ValueError(
                "SparseDeltaPairStore.put requires individual_a and individual_b"
            )
        key = pair_hash_key(doc_a, doc_b)
        with self._lock:
            if key in self._store:
                if self.evict_policy == "lru":
                    self._store.move_to_end(key)
                return

        # Canonicalize individuals to match the stored (sorted) order.
        canon_pair = canonical_pair_key(doc_a, doc_b)
        canon_a = canon_pair[0]
        if canon_a == doc_a:
            base_a, base_b = individual_a, individual_b
        else:
            base_a, base_b = individual_b, individual_a

        if len(joint_layers) != len(base_a) or len(joint_layers) != len(base_b):
            raise ValueError("layer count mismatch between joint and individuals")

        la = base_a[0][0].shape[0]
        lb = base_b[0][0].shape[0]
        total = la + lb
        if joint_layers[0][0].shape[0] != total:
            raise ValueError(
                f"joint length {joint_layers[0][0].shape[0]} != "
                f"L_a+L_b={total}"
            )

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
            # Match device/dtype of joint for the subtraction; individuals
            # may have been written to the FIFO on a different device.
            base_k = base_k.to(device=jk.device, dtype=jk.dtype)
            base_v = base_v.to(device=jv.device, dtype=jv.dtype)
            dK = jk - base_k
            dV = jv - base_v

            # Score per seq position: ‖ΔK‖_2 + ‖ΔV‖_2 over the trailing
            # feature dims.  Real vLLM hack_kv is 2-D
            # ``(seq_len, num_kv_heads * head_size)`` because ``qkv.split``
            # flattens; unit tests use 3-D.  Reduce all dims past 0 to be
            # shape-agnostic.
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
            # Race: another thread may have inserted this key while we
            # built ``deltas``.  Defer to the existing entry.
            if key in self._store:
                if self.evict_policy == "lru":
                    self._store.move_to_end(key)
                return
            self._store[key] = deltas
            self._seq_lens[key] = total
            self._pair_names[key] = canon_pair
            self._evict_to_limits_locked()

    def _evict_to_limits_locked(self) -> None:
        """See :meth:`FullJointPairStore._evict_to_limits_locked`."""
        while len(self._store) > self.max_entries:
            self._evict_one()
        if self.bytes_budget is None:
            return
        while (
            len(self._store) > 1
            and self._bytes_used_locked() > self.bytes_budget
        ):
            self._evict_one()

    def _evict_one(self) -> None:
        if self.evict_policy == "lfu":
            assert self.priority_fn is not None
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
        total = 0
        for entry in self._store.values():
            for indices, dK, dV in entry:
                total += indices.numel() * indices.element_size()
                total += dK.numel() * dK.element_size()
                total += dV.numel() * dV.element_size()
        return total
