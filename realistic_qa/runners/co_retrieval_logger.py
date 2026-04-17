"""Counts pair co-retrievals and emits promotion candidates at a threshold.

The existing ``CoRetrievalTracker`` in standard_qa/runners/utils.py covers the
post-hoc Zipf analysis; this module focuses on the online signal that drives
composition-cache promotion: "has pair (a, b) been co-retrieved >= T times?"

Each call to :meth:`record` returns the set of pairs whose count *just*
crossed the threshold on this query, so callers can enqueue them for
asynchronous precomputation.
"""
from __future__ import annotations

import collections
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from pair_kv_store import canonical_pair_key


PairKey = Tuple[str, str]


class CoRetrievalLogger:
    def __init__(self, promotion_threshold: int = 10) -> None:
        if promotion_threshold < 2:
            raise ValueError("promotion_threshold must be >= 2")
        self.promotion_threshold = promotion_threshold
        self.pair_counts: collections.Counter[PairKey] = collections.Counter()
        self.queries_seen = 0
        # Pairs already flagged for promotion (so we enqueue each exactly once).
        self._flagged: Set[PairKey] = set()

    def record(self, doc_ids: Iterable[str]) -> List[PairKey]:
        """Record one retrieval set. Return pairs that just crossed the threshold."""
        unique = sorted(set(doc_ids))
        self.queries_seen += 1
        newly_ready: List[PairKey] = []
        for a, b in combinations(unique, 2):
            key = canonical_pair_key(a, b)
            self.pair_counts[key] += 1
            if (
                self.pair_counts[key] >= self.promotion_threshold
                and key not in self._flagged
            ):
                self._flagged.add(key)
                newly_ready.append(key)
        return newly_ready

    def count(self, a: str, b: str) -> int:
        if a == b:
            return 0
        return self.pair_counts.get(canonical_pair_key(a, b), 0)

    def is_flagged(self, a: str, b: str) -> bool:
        if a == b:
            return False
        return canonical_pair_key(a, b) in self._flagged

    def save(self, path: str | Path) -> None:
        payload = {
            "promotion_threshold": self.promotion_threshold,
            "queries_seen": self.queries_seen,
            "pair_counts": [
                {"pair": list(pair), "count": int(cnt)}
                for pair, cnt in self.pair_counts.most_common()
            ],
            "flagged": [list(p) for p in sorted(self._flagged)],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.promotion_threshold = int(payload["promotion_threshold"])
        self.queries_seen = int(payload["queries_seen"])
        self.pair_counts = collections.Counter(
            {tuple(e["pair"]): int(e["count"]) for e in payload["pair_counts"]}
        )
        self._flagged = {tuple(p) for p in payload["flagged"]}

    def summary(self) -> dict:
        return {
            "promotion_threshold": self.promotion_threshold,
            "queries_seen": self.queries_seen,
            "n_unique_pairs": len(self.pair_counts),
            "n_flagged": len(self._flagged),
            "top_pairs": [
                (list(pair), int(cnt))
                for pair, cnt in self.pair_counts.most_common(20)
            ],
        }
