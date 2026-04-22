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


def test_fifo_eviction_ignores_recency():
    """With fifo=True, get()/put-existing do NOT refresh insertion order."""
    store = FullJointPairStore(max_entries=2, fifo=True)
    store.put("a", "b", _mk_layers(1))
    store.put("c", "d", _mk_layers(2))
    # "Touch" ("a","b") — in LRU mode this would save it; in FIFO it must not.
    _ = store.get("a", "b")
    store.put("a", "b", _mk_layers(1))  # no-op bump
    store.put("e", "f", _mk_layers(3))  # must evict the OLDEST insertion: ("a","b")
    assert not store.contains("a", "b")
    assert store.contains("c", "d")
    assert store.contains("e", "f")
    s = store.stats()
    assert s["fifo"] is True
    assert s["evictions"] == 1


def test_fifo_flag_surfaces_in_stats():
    assert FullJointPairStore(max_entries=1).stats()["fifo"] is False
    assert FullJointPairStore(max_entries=1, fifo=True).stats()["fifo"] is True


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


# ---------------------------------------------------------------------------
# bytes_used() — memory-savings benchmarks depend on this measurement.
# ---------------------------------------------------------------------------


def test_full_store_bytes_used_empty_is_zero():
    assert FullJointPairStore(max_entries=2).bytes_used() == 0


def test_full_store_bytes_used_matches_tensor_size():
    store = FullJointPairStore(max_entries=2)
    # 2 layers × (K+V) × 7 tokens × 8 heads × 128 dims × fp32 (4 bytes).
    layers = _mk_layers(7, n_layers=2, num_kv_heads=8, head_size=128)
    store.put("a", "b", layers)
    expected = 2 * 2 * 7 * 8 * 128 * 4
    assert store.bytes_used() == expected
    assert store.stats()["bytes_used"] == expected


def test_delta_store_bytes_used_scales_with_ratio():
    """At ratio=r the delta should carry ~r × the full bytes (plus index overhead)."""
    la, lb = 16, 16
    ind_a, ind_b, joint = _mk_individuals_and_joint(la, lb, n_layers=2)
    full = FullJointPairStore(max_entries=1)
    full.put("a", "b", joint)

    half = SparseDeltaPairStore(max_entries=1, top_k_ratio=0.5)
    half.put("a", "b", joint, individual_a=ind_a, individual_b=ind_b)

    quarter = SparseDeltaPairStore(max_entries=1, top_k_ratio=0.25)
    quarter.put("a", "b", joint, individual_a=ind_a, individual_b=ind_b)

    f = full.bytes_used()
    h = half.bytes_used()
    q = quarter.bytes_used()
    # Ratio-1.0 would exactly equal full; 0.5 → half; 0.25 → quarter
    # (modulo the int64 index overhead, which is bounded at
    # ~ K × 8 bytes per layer, negligible for these dims).
    assert q < h < f
    # The saved fraction should be close to the advertised ratio.
    assert h <= 0.6 * f   # 0.5 + overhead
    assert q <= 0.35 * f  # 0.25 + overhead


# ---------------------------------------------------------------------------
# LFU eviction — the popularity-budget runner depends on this policy.
# ---------------------------------------------------------------------------


def test_full_store_lfu_requires_priority_fn():
    with pytest.raises(ValueError):
        FullJointPairStore(max_entries=1, evict_policy="lfu")


def test_full_store_lfu_evicts_lowest_priority():
    """At capacity, the pair with the smallest priority must be dropped,
    regardless of insertion or access order."""
    priorities = {
        ("a", "b"): 10,  # most popular
        ("c", "d"): 1,   # cold → should be evicted first
        ("e", "f"): 5,
    }

    def pfn(a: str, b: str) -> float:
        return float(priorities.get(tuple(sorted((a, b))), 0))

    store = FullJointPairStore(max_entries=2, evict_policy="lfu", priority_fn=pfn)
    store.put("a", "b", _mk_layers(1))
    store.put("c", "d", _mk_layers(2))
    # Cold ("c","d") must go when ("e","f") (priority 5) arrives.
    store.put("e", "f", _mk_layers(3))
    assert store.contains("a", "b")
    assert store.contains("e", "f")
    assert not store.contains("c", "d")
    s = store.stats()
    assert s["evict_policy"] == "lfu"
    assert s["evictions"] == 1


def test_full_store_lfu_is_not_swayed_by_reads():
    """LFU: reading a cold entry must NOT save it from eviction (priority decides)."""
    priorities = {
        ("a", "b"): 0,   # always lowest
        ("c", "d"): 9,
        ("e", "f"): 9,
    }

    def pfn(a: str, b: str) -> float:
        return float(priorities.get(tuple(sorted((a, b))), 0))

    store = FullJointPairStore(max_entries=2, evict_policy="lfu", priority_fn=pfn)
    store.put("a", "b", _mk_layers(1))
    store.put("c", "d", _mk_layers(2))
    # Read cold ("a","b") many times — LRU would save it, LFU must not.
    for _ in range(5):
        _ = store.get("a", "b")
    store.put("e", "f", _mk_layers(3))
    assert not store.contains("a", "b")
    assert store.contains("c", "d")
    assert store.contains("e", "f")


def test_delta_store_lfu_evicts_lowest_priority():
    priorities = {
        ("a", "b"): 2,
        ("c", "d"): 0,
        ("e", "f"): 7,
    }

    def pfn(a: str, b: str) -> float:
        return float(priorities.get(tuple(sorted((a, b))), 0))

    store = SparseDeltaPairStore(
        max_entries=2, top_k_ratio=1.0,
        evict_policy="lfu", priority_fn=pfn,
    )
    ia1, ib1, j1 = _mk_individuals_and_joint(la=2, lb=2, seed=20)
    ia2, ib2, j2 = _mk_individuals_and_joint(la=2, lb=2, seed=21)
    ia3, ib3, j3 = _mk_individuals_and_joint(la=2, lb=2, seed=22)
    store.put("a", "b", j1, individual_a=ia1, individual_b=ib1)
    store.put("c", "d", j2, individual_a=ia2, individual_b=ib2)
    store.put("e", "f", j3, individual_a=ia3, individual_b=ib3)
    assert store.contains("a", "b")
    assert store.contains("e", "f")
    assert not store.contains("c", "d")  # lowest priority → evicted
    s = store.stats()
    assert s["evict_policy"] == "lfu"


def test_delta_store_lfu_evicted_seq_lens_cleaned_up():
    """Eviction must also drop ``_seq_lens`` / ``_pair_names`` for the victim."""
    priorities = {("a", "b"): 0, ("c", "d"): 9}

    def pfn(a: str, b: str) -> float:
        return float(priorities.get(tuple(sorted((a, b))), 0))

    store = SparseDeltaPairStore(
        max_entries=1, top_k_ratio=1.0,
        evict_policy="lfu", priority_fn=pfn,
    )
    ia1, ib1, j1 = _mk_individuals_and_joint(la=2, lb=2, seed=30)
    ia2, ib2, j2 = _mk_individuals_and_joint(la=2, lb=2, seed=31)
    store.put("a", "b", j1, individual_a=ia1, individual_b=ib1)
    store.put("c", "d", j2, individual_a=ia2, individual_b=ib2)
    # ("a","b") was evicted; its metadata must not leak.
    assert pair_hash_key("a", "b") not in store._seq_lens
    assert pair_hash_key("a", "b") not in store._pair_names


def test_evict_policy_rejects_unknown_value():
    with pytest.raises(ValueError):
        FullJointPairStore(max_entries=1, evict_policy="bogus")
    with pytest.raises(ValueError):
        SparseDeltaPairStore(max_entries=1, evict_policy="bogus")


def test_delta_store_handles_2d_hack_kv_layout():
    """Real vLLM ``self_attn.hack_kv`` is 2-D ``(seq_len, kv_size)`` because
    ``qkv.split(...)`` flattens heads × head_dim. The delta store must
    accept that layout (regression: previously hard-coded ``sum(dim=(1,2))``
    raised ``Dimension out of range`` against real KVs)."""
    la, lb = 4, 6
    kv_size = 8 * 128  # num_kv_heads × head_size, like Mistral-7B
    n_layers = 3

    def _flat(L: int, val: float):
        return [
            [
                torch.full((L, kv_size), val, dtype=torch.float32),
                torch.full((L, kv_size), -val, dtype=torch.float32),
            ]
            for _ in range(n_layers)
        ]

    ind_a = _flat(la, 1.0)
    ind_b = _flat(lb, 2.0)
    joint = [
        [
            torch.cat((ind_a[i][0], ind_b[i][0]), dim=0) + 0.05,
            torch.cat((ind_a[i][1], ind_b[i][1]), dim=0) - 0.05,
        ]
        for i in range(n_layers)
    ]
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=0.5)
    store.put("d_a", "d_b", joint, individual_a=ind_a, individual_b=ind_b)
    got = store.get("d_a", "d_b", individual_a=ind_a, individual_b=ind_b)
    assert got is not None
    assert len(got) == n_layers
    for k, v in got:
        assert k.shape == (la + lb, kv_size)
        assert v.shape == (la + lb, kv_size)
    assert store.bytes_used() > 0


# ---------------------------------------------------------------------------
# Byte-budget eviction
# ---------------------------------------------------------------------------

def _entry_bytes_full(seq_len: int, n_layers: int, num_kv_heads: int, head_size: int) -> int:
    # float32 tensors; one K + one V per layer, each seq_len * num_kv_heads * head_size.
    per_layer = 2 * seq_len * num_kv_heads * head_size * 4
    return per_layer * n_layers


def test_full_joint_bytes_budget_evicts_until_under_cap():
    n_layers, nh, hs = 2, 2, 8
    per_entry = _entry_bytes_full(seq_len=4, n_layers=n_layers, num_kv_heads=nh, head_size=hs)
    # Budget large enough for two entries, not three; max_entries leaves plenty of slack so
    # the byte budget is what actually bites.
    store = FullJointPairStore(
        max_entries=100,
        bytes_budget=int(per_entry * 2.5),
    )
    mk = lambda s: _mk_layers(s, n_layers=n_layers, num_kv_heads=nh, head_size=hs)
    store.put("a", "b", mk(4))
    store.put("c", "d", mk(4))
    assert len(store) == 2
    assert store.bytes_used() <= int(per_entry * 2.5)
    store.put("e", "f", mk(4))
    # Insertion forces eviction of the LRU entry under the byte cap.
    assert len(store) == 2
    assert not store.contains("a", "b")
    assert store.contains("c", "d")
    assert store.contains("e", "f")
    assert store.bytes_used() <= int(per_entry * 2.5)
    s = store.stats()
    assert s["bytes_budget"] == int(per_entry * 2.5)
    assert s["evictions"] >= 1


def test_full_joint_bytes_budget_keeps_at_least_one_oversize_entry():
    # Single entry larger than the entire budget must still be admitted — the store
    # never self-empties below 1 entry to avoid pathological infinite-eviction loops.
    store = FullJointPairStore(max_entries=10, bytes_budget=128)
    store.put("a", "b", _mk_layers(8))
    assert len(store) == 1
    assert store.bytes_used() > 128  # above budget, but retained


def test_full_joint_bytes_budget_respects_max_entries_cap():
    # max_entries is the hard ceiling; byte budget is softer.
    per_entry = _entry_bytes_full(seq_len=2, n_layers=2, num_kv_heads=2, head_size=4)
    store = FullJointPairStore(
        max_entries=2,
        bytes_budget=per_entry * 100,  # never trips
    )
    mk = lambda s: _mk_layers(s, n_layers=2, num_kv_heads=2, head_size=4)
    store.put("a", "b", mk(2))
    store.put("c", "d", mk(2))
    store.put("e", "f", mk(2))
    assert len(store) == 2
    assert not store.contains("a", "b")


def test_sparse_delta_bytes_budget_fits_more_entries_than_full_joint():
    # Same budget, same pairs, same seq-lens → the delta store should fit strictly
    # more entries because each delta entry is ~top_k_ratio of the joint's size
    # (plus indices overhead).  This is the property the delta-memory benchmark
    # depends on.
    n_layers, nh, hs = 2, 4, 16
    la = lb = 8

    def _ind(seed: float, L: int):
        return [
            [torch.full((L, nh, hs), seed), torch.full((L, nh, hs), seed + 0.5)]
            for _ in range(n_layers)
        ]

    def _joint(ind_a, ind_b):
        return [
            [
                torch.cat((ind_a[i][0], ind_b[i][0]), dim=0) + 0.1,
                torch.cat((ind_a[i][1], ind_b[i][1]), dim=0) - 0.1,
            ]
            for i in range(n_layers)
        ]

    pairs = [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j")]
    individuals = {name: _ind(float(idx + 1), la) for idx, name in enumerate("abcdefghij")}

    # Size one Full-Joint entry, then use 2.5× that as the shared budget.
    per_entry_full = 2 * (la + lb) * nh * hs * 4 * n_layers
    budget = int(per_entry_full * 2.5)

    full_store = FullJointPairStore(max_entries=100, bytes_budget=budget)
    delta_store = SparseDeltaPairStore(
        max_entries=100, bytes_budget=budget, top_k_ratio=0.1
    )

    for a, b in pairs:
        ind_a, ind_b = individuals[a], individuals[b]
        joint = _joint(ind_a, ind_b)
        full_store.put(a, b, joint)
        delta_store.put(a, b, joint, individual_a=ind_a, individual_b=ind_b)

    assert full_store.bytes_used() <= budget
    assert delta_store.bytes_used() <= budget
    assert len(delta_store) > len(full_store), (
        f"delta should fit more pairs; got full={len(full_store)} delta={len(delta_store)}"
    )


def test_sparse_delta_bytes_budget_lfu_eviction_drops_lowest_priority():
    # Under LFU, a fixed byte budget must evict the *lowest-priority* pair, not
    # the oldest.  Uses a tight budget that only fits two entries.
    n_layers, nh, hs = 2, 2, 4
    la = lb = 4
    ind = lambda seed: [
        [torch.full((la, nh, hs), seed), torch.full((la, nh, hs), seed + 0.25)]
        for _ in range(n_layers)
    ]
    joint = lambda a, b: [
        [
            torch.cat((a[i][0], b[i][0]), dim=0) + 0.1,
            torch.cat((a[i][1], b[i][1]), dim=0) - 0.1,
        ]
        for i in range(n_layers)
    ]

    popularity = {
        ("a", "b"): 3,
        ("c", "d"): 1,  # least popular → evicted first
        ("e", "f"): 5,
    }

    def prio(x: str, y: str) -> float:
        return float(popularity.get(canonical_pair_key(x, y), 0))

    # Full-joint per-entry size; delta entry (top_k_ratio=0.5) is ~half of that.
    # Sizing the budget to ~1.2 × per_entry leaves just enough room for 2 delta
    # entries, so the third insertion must evict exactly one pair.
    per_entry = 2 * (la + lb) * nh * hs * 4 * n_layers
    store = SparseDeltaPairStore(
        max_entries=100,
        bytes_budget=int(per_entry * 1.2),
        top_k_ratio=0.5,
        evict_policy="lfu",
        priority_fn=prio,
    )

    inds = {name: ind(float(i + 1)) for i, name in enumerate("abcdef")}
    for a, b in [("a", "b"), ("c", "d"), ("e", "f")]:
        store.put(a, b, joint(inds[a], inds[b]), individual_a=inds[a], individual_b=inds[b])

    assert len(store) == 2, f"expected 2 entries under byte budget, got {len(store)}"
    assert store.contains("a", "b")
    assert store.contains("e", "f")
    assert not store.contains("c", "d"), (
        "LFU under byte budget must drop the lowest-priority pair first"
    )
