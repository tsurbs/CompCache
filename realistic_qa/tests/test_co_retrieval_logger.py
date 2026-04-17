from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from co_retrieval_logger import CoRetrievalLogger  # noqa: E402


def test_threshold_triggers_once():
    log = CoRetrievalLogger(promotion_threshold=3)
    assert log.record(["d1", "d2"]) == []
    assert log.record(["d1", "d2"]) == []
    newly = log.record(["d1", "d2"])
    # First crossing: expect exactly this pair.
    assert newly == [("d1", "d2")]
    # Subsequent co-retrievals: no duplicate enqueue.
    assert log.record(["d1", "d2"]) == []
    assert log.record(["d1", "d2", "d3"]) == []
    # Counts keep incrementing though.
    assert log.count("d1", "d2") == 5


def test_multiple_new_pairs_in_one_query():
    log = CoRetrievalLogger(promotion_threshold=2)
    log.record(["a", "b", "c"])
    newly = log.record(["a", "b", "c"])
    # All three pairs crossed 2 on this second query.
    assert sorted(newly) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_canonical_regardless_of_input_order():
    log = CoRetrievalLogger(promotion_threshold=2)
    log.record(["z", "a"])
    newly = log.record(["a", "z"])
    assert newly == [("a", "z")]
    assert log.count("z", "a") == 2


def test_duplicates_in_single_query_counted_once():
    log = CoRetrievalLogger(promotion_threshold=2)
    log.record(["a", "a", "b"])  # unique = {a,b} → one pair, count 1
    assert log.count("a", "b") == 1
    newly = log.record(["a", "b"])
    assert newly == [("a", "b")]


def test_save_and_load_round_trip(tmp_path: Path):
    log = CoRetrievalLogger(promotion_threshold=3)
    for _ in range(3):
        log.record(["a", "b"])
    log.record(["a", "c"])
    p = tmp_path / "state.json"
    log.save(p)

    restored = CoRetrievalLogger()
    restored.load(p)
    assert restored.promotion_threshold == 3
    assert restored.queries_seen == 4
    assert restored.count("a", "b") == 3
    assert restored.count("a", "c") == 1
    # ("a","b") was flagged before save; must stay flagged after load.
    assert restored.is_flagged("a", "b")
    # Re-crossing the threshold must NOT re-emit a flagged pair.
    assert restored.record(["a", "b"]) == []


def test_rejects_bad_threshold():
    with pytest.raises(ValueError):
        CoRetrievalLogger(promotion_threshold=1)
