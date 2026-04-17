"""Maximum-coverage disjoint pair matching over a retrieved chunk list.

For a retrieved set of document IDs and a membership predicate that reports
whether a given pair has a cached joint KV, return the subset of pair matches
that maximizes (1) number of chunks covered, then (2) total promotion
frequency. Each chunk belongs to at most one matched pair.

Pair counts for retrieval are small (typically k <= 10), so a simple
branch-and-bound search over remaining indices is well within budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from pair_kv_store import canonical_pair_key


@dataclass(frozen=True)
class PairMatch:
    """A matched pair, keyed canonically and with original-order positions.

    ``positions[0]`` is the original index of the canonically-first doc id,
    ``positions[1]`` of the second. That means reordering each pair's
    members to ``positions[0]`` then ``positions[1]`` yields canonical order.
    """

    pair_key: Tuple[str, str]
    positions: Tuple[int, int]


def find_best_matching(
    doc_ids: Sequence[str],
    is_cached: Callable[[str, str], bool],
    pair_frequency: Callable[[str, str], int] = lambda a, b: 0,
) -> List[PairMatch]:
    """Return matches that maximize coverage, tie-breaking by total frequency."""
    n = len(doc_ids)
    if n < 2:
        return []

    # Precompute edges between positions whose doc pair is cached.
    edges: dict[int, list[int]] = {i: [] for i in range(n)}
    edge_freq: dict[Tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if doc_ids[i] == doc_ids[j]:
                # Same doc retrieved twice: cannot form a self-pair.
                continue
            if is_cached(doc_ids[i], doc_ids[j]):
                edges[i].append(j)
                edges[j].append(i)
                edge_freq[(i, j)] = pair_frequency(doc_ids[i], doc_ids[j])

    best: List[PairMatch] = []
    best_score = (0, 0)  # (coverage, total_freq)

    def _build_match(i: int, j: int) -> PairMatch:
        lo, hi = (i, j) if i < j else (j, i)
        key = canonical_pair_key(doc_ids[lo], doc_ids[hi])
        # Position order must match canonical key order.
        if key[0] == doc_ids[lo]:
            positions = (lo, hi)
        else:
            positions = (hi, lo)
        return PairMatch(pair_key=key, positions=positions)

    def _search(available: frozenset[int], acc: List[PairMatch], acc_freq: int) -> None:
        nonlocal best, best_score
        # Upper bound: current coverage + 2 * floor(remaining/2). Prunes hopeless branches.
        upper = 2 * len(acc) + 2 * (len(available) // 2)
        if (upper, acc_freq) <= best_score:
            return

        if not available:
            score = (2 * len(acc), acc_freq)
            if score > best_score:
                best_score = score
                best = list(acc)
            return

        i = min(available)
        rest = available - {i}

        # Branch: do not match i.
        _search(rest, acc, acc_freq)

        # Branch: match i with each available partner.
        for j in edges[i]:
            if j not in rest:
                continue
            freq = edge_freq[(i, j) if i < j else (j, i)]
            acc.append(_build_match(i, j))
            _search(rest - {j}, acc, acc_freq + freq)
            acc.pop()

    _search(frozenset(range(n)), [], 0)
    # Handle the empty-match degenerate case (no edges): best stays [] with score (0,0).
    return best


def apply_reordering(
    doc_ids: Sequence[str], matches: Sequence[PairMatch]
) -> List[int]:
    """Return a permutation of original indices honoring within-pair canonical order.

    Non-pair chunks keep their relative position. Matched pairs are emitted
    adjacent in canonical order at the first index (in retrieval order) where
    either of their members appears.
    """
    n = len(doc_ids)
    pos_to_match: dict[int, PairMatch] = {}
    for m in matches:
        pos_to_match[m.positions[0]] = m
        pos_to_match[m.positions[1]] = m

    emitted: set[int] = set()
    order: List[int] = []
    for idx in range(n):
        if idx in emitted:
            continue
        m = pos_to_match.get(idx)
        if m is None:
            order.append(idx)
            emitted.add(idx)
            continue
        a, b = m.positions  # canonical order
        order.append(a)
        order.append(b)
        emitted.add(a)
        emitted.add(b)
    return order
