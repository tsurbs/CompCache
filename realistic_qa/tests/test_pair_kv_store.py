from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from pair_kv_store import (  # noqa: E402
    FullJointPairStore,
    SparseDeltaPairStore,
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


# ---------------------------------------------------------------------------
# SparseDeltaPairStore (§3.2)
# ---------------------------------------------------------------------------


def _mk_individuals_and_joint(
    la: int,
    lb: int,
    *,
    n_layers: int = 2,
    num_kv_heads: int = 8,
    head_size: int = 128,
    seed: int = 0,
):
    """Make realistic-looking individual KVs plus a joint equal to cat(a, b)
    + a structured perturbation, so the delta has a well-defined top-K."""
    torch.manual_seed(seed)
    ind_a = [
        [
            torch.randn(la, num_kv_heads, head_size),
            torch.randn(la, num_kv_heads, head_size),
        ]
        for _ in range(n_layers)
    ]
    ind_b = [
        [
            torch.randn(lb, num_kv_heads, head_size),
            torch.randn(lb, num_kv_heads, head_size),
        ]
        for _ in range(n_layers)
    ]
    # Build joint = cat(ind_a, ind_b) + perturbation whose magnitude varies per row.
    joint = []
    for layer_idx in range(n_layers):
        base_k = torch.cat((ind_a[layer_idx][0], ind_b[layer_idx][0]), dim=0)
        base_v = torch.cat((ind_a[layer_idx][1], ind_b[layer_idx][1]), dim=0)
        total = la + lb
        # Row scaling so score per position is monotone and distinct.
        scale = torch.linspace(0.1, 1.0, total).view(total, 1, 1)
        pert_k = torch.randn_like(base_k) * scale
        pert_v = torch.randn_like(base_v) * scale
        joint.append([base_k + pert_k, base_v + pert_v])
    return ind_a, ind_b, joint


def test_delta_store_round_trip_ratio_one_is_exact():
    """At top_k_ratio=1.0 every position is kept, so reconstruction is exact."""
    store = SparseDeltaPairStore(max_entries=4, top_k_ratio=1.0)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=5, lb=7)
    store.put("a", "b", joint, individual_a=ind_a, individual_b=ind_b)
    got = store.get("a", "b", individual_a=ind_a, individual_b=ind_b)
    assert got is not None
    for (jk, jv), (gk, gv) in zip(joint, got):
        assert torch.allclose(jk, gk, atol=1e-6)
        assert torch.allclose(jv, gv, atol=1e-6)


def test_delta_store_needs_individuals_flag():
    assert SparseDeltaPairStore.needs_individuals is True
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    assert store.needs_individuals is True


def test_delta_store_put_requires_individuals():
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=3, lb=3)
    with pytest.raises(ValueError):
        store.put("a", "b", joint)  # neither individual provided
    with pytest.raises(ValueError):
        store.put("a", "b", joint, individual_a=ind_a)  # only one


def test_delta_store_get_without_individuals_is_miss():
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=3, lb=3)
    store.put("a", "b", joint, individual_a=ind_a, individual_b=ind_b)
    # No individuals passed → treated as miss (can't reconstruct).
    assert store.get("a", "b") is None
    s = store.stats()
    assert s["misses"] == 1
    assert s["hits"] == 0


def test_delta_store_canonical_order_invariance():
    """put/get must return the same joint regardless of (a,b) vs (b,a) order,
    after canonicalization. Tests that individuals are reassigned correctly."""
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=4, lb=6, seed=1)
    # "apple" < "banana" canonically.
    store.put("apple", "banana", joint, individual_a=ind_a, individual_b=ind_b)
    assert store.contains("banana", "apple")
    # Reversed-order get passes individuals in reversed order too.
    got = store.get("banana", "apple", individual_a=ind_b, individual_b=ind_a)
    assert got is not None
    for (jk, jv), (gk, gv) in zip(joint, got):
        assert torch.allclose(jk, gk, atol=1e-6)
        assert torch.allclose(jv, gv, atol=1e-6)


def test_delta_store_sparse_keeps_top_k_per_layer():
    """With ratio=0.25 on total=12, K=3 positions per layer are retained."""
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=0.25)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=5, lb=7, seed=2)
    store.put("a", "b", joint, individual_a=ind_a, individual_b=ind_b)
    # Internals: one entry per layer, each a 3-tuple (indices, dK, dV).
    entry = store._store[pair_hash_key("a", "b")]
    for indices, dK, dV in entry:
        assert indices.shape == (3,)
        assert dK.shape[0] == 3
        assert dV.shape[0] == 3
    # Reconstruction will not be exact (some rows dropped); but the delta
    # between joint and reconstruction must be <= the delta between joint
    # and the raw cat-of-individuals baseline (top-K by definition captures
    # the highest-magnitude perturbation).
    got = store.get("a", "b", individual_a=ind_a, individual_b=ind_b)
    assert got is not None
    for layer_idx, ((jk, jv), (gk, gv)) in enumerate(zip(joint, got)):
        baseline_k = torch.cat((ind_a[layer_idx][0], ind_b[layer_idx][0]), dim=0)
        baseline_v = torch.cat((ind_a[layer_idx][1], ind_b[layer_idx][1]), dim=0)
        err_reconstructed = (jk - gk).norm() + (jv - gv).norm()
        err_baseline = (jk - baseline_k).norm() + (jv - baseline_v).norm()
        assert err_reconstructed <= err_baseline + 1e-5


def test_delta_store_rejects_bad_ratio():
    with pytest.raises(ValueError):
        SparseDeltaPairStore(max_entries=2, top_k_ratio=0.0)
    with pytest.raises(ValueError):
        SparseDeltaPairStore(max_entries=2, top_k_ratio=1.5)


def test_delta_store_length_mismatch_raises():
    """Joint shape must equal L_a + L_b along seq dim."""
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    ind_a, ind_b, joint = _mk_individuals_and_joint(la=3, lb=5)
    # Corrupt joint to have the wrong length.
    bad_joint = [[j[0][:-1].clone(), j[1][:-1].clone()] for j in joint]
    with pytest.raises(ValueError):
        store.put("a", "b", bad_joint, individual_a=ind_a, individual_b=ind_b)


def test_delta_store_lru_eviction():
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    ia1, ib1, j1 = _mk_individuals_and_joint(la=2, lb=2, seed=10)
    ia2, ib2, j2 = _mk_individuals_and_joint(la=2, lb=2, seed=11)
    ia3, ib3, j3 = _mk_individuals_and_joint(la=2, lb=2, seed=12)
    store.put("a", "b", j1, individual_a=ia1, individual_b=ib1)
    store.put("c", "d", j2, individual_a=ia2, individual_b=ib2)
    # Touch ("a","b") so ("c","d") becomes LRU.
    _ = store.get("a", "b", individual_a=ia1, individual_b=ib1)
    store.put("e", "f", j3, individual_a=ia3, individual_b=ib3)
    assert store.contains("a", "b")
    assert not store.contains("c", "d")
    assert store.contains("e", "f")
    s = store.stats()
    assert s["entries"] == 2
    assert s["evictions"] == 1


def test_delta_store_stats_fields():
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=0.5)
    s = store.stats()
    assert s["kind"] == "sparse_delta"
    assert s["top_k_ratio"] == 0.5
    assert s["capacity"] == 2
    assert "reconstruct_failures" in s
