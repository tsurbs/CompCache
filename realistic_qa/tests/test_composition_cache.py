from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from co_retrieval_logger import CoRetrievalLogger  # noqa: E402
from composition_cache import CompositionCache  # noqa: E402
from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402
from pair_kv_store import FullJointPairStore  # noqa: E402
from promotion_worker import PromotionWorker  # noqa: E402


N_LAYERS = 2
N_HEADS = 2
HEAD_DIM = 4


def _layers_for(n_tokens: int, marker: float) -> list[list[torch.Tensor]]:
    """Per-layer [K, V] filled with a sentinel so tests can tell chunks apart."""
    return [
        [
            torch.full((n_tokens, N_HEADS, HEAD_DIM), marker, dtype=torch.float32),
            torch.full((n_tokens, N_HEADS, HEAD_DIM), -marker, dtype=torch.float32),
        ]
        for _ in range(N_LAYERS)
    ]


def _make_instance(*, pair_cap: int = 4, fifo_cap: int = 16, threshold: int = 2,
                   worker: PromotionWorker | None = None, promote_sync: bool = False):
    individual = FIFOChunkKVCache(max_entries=fifo_cap)
    store = FullJointPairStore(max_entries=pair_cap)
    logger = CoRetrievalLogger(promotion_threshold=threshold)
    cc = CompositionCache(
        individual_cache=individual,
        pair_store=store,
        logger=logger,
        promotion_worker=worker,
        promote_sync=promote_sync,
    )
    return cc, individual, store, logger


def _chunk_cache_key(tokens: List[int]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.to_bytes(4, "little", signed=True))
    return f"chunk:{h.hexdigest()}"


def _make_forwards(instr_marker: float = 100.0):
    """Build forward callbacks that return sentinel KVs keyed by token content."""
    calls = {"instr": 0, "chunk": []}

    def run_instr(token_ids, slice_start, slice_end):
        calls["instr"] += 1
        calls["instr_slice"] = (slice_start, slice_end)
        return _layers_for(len(token_ids), instr_marker)

    def run_chunk(token_ids, slice_start, slice_end):
        calls["chunk"].append((tuple(token_ids), slice_start, slice_end))
        # Use first token as sentinel marker so different chunks produce different KVs.
        marker = float(token_ids[0]) if token_ids else 1.0
        return _layers_for(len(token_ids), marker)

    return run_instr, run_chunk, calls


# ---- Cold start: all individual misses ------------------------------------


def test_cold_start_runs_all_forwards_and_populates_fifo():
    cc, individual, _, _ = _make_instance()
    instr_tokens = [9, 9, 9]
    chunks = [[10, 11], [20, 21, 22], [30]]
    query_tokens = [40, 41]
    doc_chunk_ids = [instr_tokens, *chunks, query_tokens]
    retrieval_ids = ["d_a", "d_b", "d_c"]

    run_instr, run_chunk, calls = _make_forwards()

    input_ids, kvs, reordered, stats = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=3,
        s_start_1_len=2,
    )

    # 1 instr forward + 3 chunk forwards + 1 query forward.
    assert calls["instr"] == 1
    assert len(calls["chunk"]) == 4
    assert stats.individual_misses == 5  # instr + 3 chunks + query
    assert stats.individual_hits == 0
    assert stats.pair_hits == 0
    assert stats.pair_misses == 0

    # instr slice uses s_start_len; chunk slices use s_start_1_len..len+1.
    assert calls["instr_slice"] == (0, 3)
    chunk_slices = [c for c in calls["chunk"] if c[0] != tuple(query_tokens)]
    for toks, ss, se in chunk_slices:
        assert ss == 2
        assert se == len(toks) + 1

    # Reordered chunks: no matches, so identity over doc positions.
    assert reordered == [instr_tokens, chunks[0], chunks[1], chunks[2], query_tokens]

    # input_ids strips s_start_1_len-1 = 1 from every non-first chunk.
    expected = list(instr_tokens)
    for c in [chunks[0], chunks[1], chunks[2], query_tokens]:
        expected.extend(c[1:])
    assert input_ids == expected

    # KVs: N_LAYERS layers, each concatenated over all pieces.
    total_len = len(instr_tokens) + sum(len(c) for c in chunks) + len(query_tokens)
    assert len(kvs) == N_LAYERS
    for k, v in kvs:
        assert k.shape == (total_len, N_HEADS, HEAD_DIM)
        assert v.shape == (total_len, N_HEADS, HEAD_DIM)

    # FIFO populated with instr + 3 chunks + query.
    assert len(individual) == 5


# ---- Warm start: everything hits the FIFO ---------------------------------


def test_individual_hits_skip_forwards():
    cc, _, _, _ = _make_instance()
    instr = [9, 9, 9]
    chunks = [[10, 11], [20, 21]]
    query = [40]
    doc_chunk_ids = [instr, *chunks, query]
    retrieval_ids = ["d_a", "d_b"]

    run_instr, run_chunk, calls_first = _make_forwards()
    cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=3,
        s_start_1_len=2,
    )

    # Second pass (same keys, different query key) should reuse instr + both chunks.
    run_instr2, run_chunk2, calls2 = _make_forwards()
    _, _, _, stats = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:2",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr2,
        run_chunk_forward=run_chunk2,
        s_start_len=3,
        s_start_1_len=2,
    )
    assert calls2["instr"] == 0
    # Only the new query tokens forced a miss.
    assert len(calls2["chunk"]) == 1
    assert stats.individual_hits == 3  # instr + 2 chunks
    assert stats.individual_misses == 1  # query only


# ---- Pair hit: joint KV replaces two individual forwards ------------------


def test_pair_hit_consumes_joint_kv_and_reorders():
    cc, _, store, _ = _make_instance()
    # Canonical order puts "d_a" before "d_z". But retrieval returns them swapped.
    instr = [9]
    chunk_z = [70, 71]           # retrieval position 0 → "d_z"
    chunk_a = [10, 11, 12]       # retrieval position 1 → "d_a"
    chunk_x = [50]               # retrieval position 2 → "d_x" (non-pair)
    query = [40]
    doc_chunk_ids = [instr, chunk_z, chunk_a, chunk_x, query]
    retrieval_ids = ["d_z", "d_a", "d_x"]

    # Pre-populate pair (canonical order: a before z), joint KV marked 999.
    joint = _layers_for(len(chunk_a) + len(chunk_z), 999.0)
    store.put("d_a", "d_z", joint)

    run_instr, run_chunk, calls = _make_forwards()
    _, kvs, reordered, stats = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=1,
        s_start_1_len=2,
    )

    assert stats.pair_hits == 1
    assert stats.pair_misses == 0
    # Pair members are NOT run individually. Only instr + d_x + query => 3 forwards.
    assert calls["instr"] == 1
    assert len(calls["chunk"]) == 2  # d_x + query; pair members skipped
    chunk_tokens_run = {c[0] for c in calls["chunk"]}
    assert tuple(chunk_z) not in chunk_tokens_run
    assert tuple(chunk_a) not in chunk_tokens_run

    # Reordered: instr, then chunk_a (canonical first), chunk_z, then chunk_x, then query.
    assert reordered == [instr, chunk_a, chunk_z, chunk_x, query]
    assert stats.matched_pairs == [("d_a", "d_z")]

    # Verify the joint-KV region (rows s_instr .. s_instr + len_a + len_z) is the 999 sentinel.
    len_instr = len(instr)
    len_pair = len(chunk_a) + len(chunk_z)
    for k, _v in kvs:
        pair_slice = k[len_instr : len_instr + len_pair]
        assert torch.all(pair_slice == 999.0)


# ---- Pair miss: shape mismatch falls back to individual -------------------


def test_pair_shape_mismatch_falls_back_to_individual():
    cc, _, store, _ = _make_instance()
    instr = [9]
    chunk_a = [10, 11]
    chunk_b = [20, 21, 22]
    query = [40]
    doc_chunk_ids = [instr, chunk_a, chunk_b, query]
    retrieval_ids = ["d_a", "d_b"]

    # Store a bogus joint KV whose seq_len is WRONG.
    bogus = _layers_for(999, 7.0)
    store.put("d_a", "d_b", bogus)

    run_instr, run_chunk, calls = _make_forwards()
    _, _, _, stats = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=1,
        s_start_1_len=2,
    )
    assert stats.pair_hits == 0
    assert stats.pair_misses == 1
    # Both chunks got forwarded individually + instr + query = 4 chunk calls.
    assert len(calls["chunk"]) == 3  # d_a, d_b, query (instr is separate)


# ---- Promotion enqueue on threshold crossing ------------------------------


def test_threshold_crossing_enqueues_promotion_job():
    # Use a sync worker with a stub forward so we can observe the promotion.
    store = FullJointPairStore(max_entries=4)

    promoted: list = []

    def stub_forward(concat_tokens):
        promoted.append(list(concat_tokens))
        return _layers_for(len(concat_tokens), 42.0)

    lock = threading.Lock()
    worker = PromotionWorker(store, stub_forward, lock)

    individual = FIFOChunkKVCache(max_entries=16)
    logger = CoRetrievalLogger(promotion_threshold=2)
    cc = CompositionCache(
        individual_cache=individual,
        pair_store=store,
        logger=logger,
        promotion_worker=worker,
        promote_sync=True,  # inline so no thread needed
    )

    instr = [9]
    chunk_a = [10, 11]
    chunk_b = [20, 21]
    query = [40]
    doc_chunk_ids = [instr, chunk_a, chunk_b, query]
    retrieval_ids = ["d_a", "d_b"]

    run_instr, run_chunk, _ = _make_forwards()
    # First query: pair count goes to 1, no crossing yet.
    _, _, _, stats1 = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=1,
        s_start_1_len=2,
    )
    assert stats1.promotions_enqueued == 0
    assert not store.contains("d_a", "d_b")

    # Second query: threshold of 2 is crossed → promote.
    _, _, _, stats2 = cc.assemble(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:2",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=1,
        s_start_1_len=2,
    )
    assert stats2.promotions_enqueued == 1
    assert store.contains("d_a", "d_b")
    # Worker received the full concatenation in canonical pair order (a first).
    assert promoted == [chunk_a + chunk_b]


def test_newly_ready_pair_not_in_current_retrieval_is_skipped():
    store = FullJointPairStore(max_entries=4)
    promoted: list = []

    def stub_forward(tokens):
        promoted.append(list(tokens))
        return _layers_for(len(tokens), 1.0)

    worker = PromotionWorker(store, stub_forward, threading.Lock())
    individual = FIFOChunkKVCache(max_entries=16)
    logger = CoRetrievalLogger(promotion_threshold=2)
    cc = CompositionCache(
        individual_cache=individual,
        pair_store=store,
        logger=logger,
        promotion_worker=worker,
        promote_sync=True,
    )

    run_instr, run_chunk, _ = _make_forwards()
    common = dict(
        instr_cache_key="__instr__",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        s_start_len=1,
        s_start_1_len=2,
    )

    # Two co-retrievals of (d_a, d_b) → flagged on the second, but we arrange
    # so that the second co-retrieval also contains a 3rd pair whose crossing
    # we want to suppress mid-query via a missing chunk id. Simpler: have the
    # second query's pair_ids include d_a + d_b normally → flagged there; we
    # then issue a third query that only retrieves d_c,d_d (the flagged pair
    # from past is no longer in the current retrieval), but we also simulate
    # that scenario by starting fresh with `_flagged` preloaded.

    # Easier direct setup: manually flag the pair in logger and hand a query
    # that does not contain it.
    logger._flagged.add(("d_a", "d_b"))
    # Force newly_ready to include that pair by also recording two queries
    # for (d_x, d_y) with threshold 2.

    # Simpler: bypass by directly appending to newly_ready via log.record? We
    # instead check the behavior by issuing a query where record() returns a
    # flagged-but-absent pair.
    # record() only returns pairs whose count JUST crossed, so we stage:
    #   query1: [d_x, d_y]  → count(x,y)=1
    # now run assemble on [d_u, d_v] where we manually bump count(x,y) via
    # logger.pair_counts to T-1, then on the next assemble the co-retrieval
    # of x,y is NOT in retrieval but also won't appear in newly_ready.
    # So the "not in retrieval" skip is only triggered if newly_ready returns
    # a pair whose tokens cannot be looked up. We force that by constructing
    # newly_ready via a synthetic record.

    # Direct test: monkeypatch logger.record to return a fabricated newly_ready.
    synthetic = [("d_missing_a", "d_missing_b")]
    cc.logger.record = lambda doc_ids: synthetic  # type: ignore[assignment]

    instr = [9]
    chunk_a = [10]
    chunk_b = [20]
    query = [40]
    _, _, _, stats = cc.assemble(
        doc_chunk_ids=[instr, chunk_a, chunk_b, query],
        retrieval_doc_ids=["d_a", "d_b"],  # neither matches synthetic pair
        query_cache_key="q:1",
        **common,
    )
    # No promotion runs because tokens for the flagged pair aren't in retrieval.
    assert stats.promotions_enqueued == 0
    assert promoted == []
    assert not store.contains("d_missing_a", "d_missing_b")
