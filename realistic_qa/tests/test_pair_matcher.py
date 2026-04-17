from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from pair_matcher import PairMatch, apply_reordering, find_best_matching  # noqa: E402


def _membership(pairs):
    """Build an is_cached(a,b) predicate over a set of canonical pairs."""
    canon = {tuple(sorted(p)) for p in pairs}

    def is_cached(a: str, b: str) -> bool:
        return tuple(sorted((a, b))) in canon

    return is_cached


def test_no_cached_pairs_returns_empty():
    doc_ids = ["d1", "d2", "d3"]
    matches = find_best_matching(doc_ids, _membership([]))
    assert matches == []


def test_single_pair_hit():
    doc_ids = ["d1", "d2", "d3"]
    matches = find_best_matching(doc_ids, _membership([("d1", "d3")]))
    assert len(matches) == 1
    m = matches[0]
    assert m.pair_key == ("d1", "d3")
    # d1 is at position 0, d3 at position 2. Canonical-first = d1 → position 0.
    assert m.positions == (0, 2)


def test_order_invariant_lookup():
    # Reversed retrieval order: d3 before d1. Positions must still line up with canonical order.
    doc_ids = ["d3", "d2", "d1"]
    matches = find_best_matching(doc_ids, _membership([("d3", "d1")]))
    assert len(matches) == 1
    m = matches[0]
    assert m.pair_key == ("d1", "d3")
    # Canonical-first is d1 at position 2; second is d3 at position 0.
    assert m.positions == (2, 0)


def test_max_coverage_picks_two_disjoint_over_one():
    # Overlap: (d1,d2) and (d2,d3) are both cached, plus disjoint (d3,d4).
    doc_ids = ["d1", "d2", "d3", "d4"]
    cached = [("d1", "d2"), ("d2", "d3"), ("d3", "d4")]
    # The optimal disjoint set is either {(d1,d2),(d3,d4)} or {(d2,d3) + nothing}.
    # Maximum coverage: 4 vs 2. Matcher must pick the 4-coverage plan.
    matches = find_best_matching(doc_ids, _membership(cached))
    assert len(matches) == 2
    keys = sorted(m.pair_key for m in matches)
    assert keys == [("d1", "d2"), ("d3", "d4")]


def test_tie_break_by_frequency():
    doc_ids = ["d1", "d2", "d3"]
    # Both (d1,d2) and (d2,d3) are cached; only one can be picked. Pick higher-frequency one.
    cached = [("d1", "d2"), ("d2", "d3")]
    freq = {("d1", "d2"): 5, ("d2", "d3"): 10}
    matches = find_best_matching(
        doc_ids,
        _membership(cached),
        pair_frequency=lambda a, b: freq[tuple(sorted((a, b)))],
    )
    assert len(matches) == 1
    assert matches[0].pair_key == ("d2", "d3")


def test_reordering_keeps_unmatched_in_place():
    doc_ids = ["d1", "d2", "d3", "d4"]
    matches = [PairMatch(pair_key=("d2", "d4"), positions=(1, 3))]
    order = apply_reordering(doc_ids, matches)
    # d1 at 0 stays. d2 emits at its first occurrence, partner d4 right after. d3 follows.
    assert order == [0, 1, 3, 2]
    assert [doc_ids[i] for i in order] == ["d1", "d2", "d4", "d3"]


def test_reordering_with_canonical_swap():
    doc_ids = ["d4", "d2", "d1", "d3"]
    # Pair (d2, d4) canonical: d2 first, d4 second. Original positions: d2=1, d4=0.
    # Canonical order means we emit position 1 then position 0 at their first-seen index.
    matches = [PairMatch(pair_key=("d2", "d4"), positions=(1, 0))]
    order = apply_reordering(doc_ids, matches)
    # First seen pair member in retrieval order is position 0 (d4). We emit canonical pair: 1, 0.
    assert order == [1, 0, 2, 3]
    assert [doc_ids[i] for i in order] == ["d2", "d4", "d1", "d3"]


def test_reordering_multiple_pairs():
    doc_ids = ["d1", "d2", "d3", "d4", "d5"]
    matches = [
        PairMatch(pair_key=("d1", "d3"), positions=(0, 2)),
        PairMatch(pair_key=("d4", "d5"), positions=(3, 4)),
    ]
    order = apply_reordering(doc_ids, matches)
    # Expected: [d1, d3, d2, d4, d5] - d1,d3 emitted adjacent; d2 in place; d4,d5 emitted adjacent.
    assert order == [0, 2, 1, 3, 4]


def test_duplicate_doc_ids_are_never_self_paired():
    # Pathological: same doc retrieved twice. No pair is cached because pair_kv_store
    # rejects self-pairs, but the matcher should also refuse internally.
    doc_ids = ["d1", "d1", "d2"]
    matches = find_best_matching(doc_ids, _membership([("d1", "d2")]))
    # One of the d1's can pair with d2 — the other is unmatched.
    assert len(matches) == 1
    assert matches[0].pair_key == ("d1", "d2")
