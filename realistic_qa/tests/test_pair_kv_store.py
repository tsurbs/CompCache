from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from pair_kv_store import (  # noqa: E402
    FullJointPairStore,
    canonical_pair_key,
    pair_hash_key,
)


def _mk_layers(seq_len: int, n_layers: int = 2, num_kv_heads: int = 8, head_size: int = 128):
    torch.manual_seed(seq_len)
    return [
        [
            torch.randn(seq_len, num_kv_heads, head_size),
            torch.randn(seq_len, num_kv_heads, head_size),
        ]
        for _ in range(n_layers)
    ]


def test_canonical_key_is_order_invariant():
    assert canonical_pair_key("a", "b") == canonical_pair_key("b", "a")
    assert pair_hash_key("a", "b") == pair_hash_key("b", "a")


def test_canonical_key_rejects_self_pair():
    with pytest.raises(ValueError):
        canonical_pair_key("a", "a")


def test_distinct_pairs_have_distinct_keys():
    # Simple sanity; not a collision proof, but enough to catch bugs.
    k1 = pair_hash_key("a", "b")
    k2 = pair_hash_key("a", "c")
    k3 = pair_hash_key("b", "c")
    assert len({k1, k2, k3}) == 3


def test_put_get_round_trip():
    store = FullJointPairStore(max_entries=4)
    layers = _mk_layers(7)
    assert not store.contains("a", "b")
    store.put("a", "b", layers)
    assert store.contains("a", "b")
    assert store.contains("b", "a")  # order-invariant
    got = store.get("b", "a")
    assert got is not None
    for (k, v), (k2, v2) in zip(layers, got):
        assert torch.equal(k, k2)
        assert torch.equal(v, v2)


def test_get_returns_clone_not_alias():
    store = FullJointPairStore(max_entries=4)
    layers = _mk_layers(3)
    store.put("a", "b", layers)
    got = store.get("a", "b")
    assert got is not None
    got[0][0].fill_(42.0)
    # Subsequent get() must still see the original values.
    got2 = store.get("a", "b")
    assert got2 is not None
    assert not torch.equal(got[0][0], got2[0][0])


def test_lru_eviction_order():
    store = FullJointPairStore(max_entries=2)
    store.put("a", "b", _mk_layers(1))
    store.put("c", "d", _mk_layers(2))
    # Touch ("a","b") so ("c","d") is the LRU.
    _ = store.get("a", "b")
    store.put("e", "f", _mk_layers(3))  # evict ("c","d")
    assert store.contains("a", "b")
    assert not store.contains("c", "d")
    assert store.contains("e", "f")
    s = store.stats()
    assert s["entries"] == 2
    assert s["evictions"] == 1
    assert s["hits"] == 1
    assert s["misses"] == 0


def test_put_duplicate_does_not_double_store():
    store = FullJointPairStore(max_entries=2)
    layers = _mk_layers(1)
    store.put("a", "b", layers)
    store.put("a", "b", layers)
    assert len(store) == 1


def test_miss_counts():
    store = FullJointPairStore(max_entries=2)
    assert store.get("a", "b") is None
    s = store.stats()
    assert s["misses"] == 1
    assert s["hits"] == 0
