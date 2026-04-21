"""Unit tests for per_query_pair_assembler.assemble_pairs_per_query.

Used by the standard_qa 3-way runner. Verifies:
- Pairs are formed greedily adjacent ([(d0,d1), (d2,d3), ...]).
- Every pair runs exactly one joint pair forward (run_pair_forward).
- No persistent caching between queries (every query recomputes).
- Odd-length retrievals fall back to per-chunk forward for the leftover.
- pair_hits == n_pairs; individual counts reflect instr + query (+ odd-out).
- input_ids reflect the original retrieval order (no reordering).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402
from per_query_pair_assembler import assemble_pairs_per_query  # noqa: E402


N_LAYERS = 2
N_HEADS = 2
HEAD_DIM = 4


def _layers_for(n_tokens: int, marker: float) -> list[list[torch.Tensor]]:
    return [
        [
            torch.full((n_tokens, N_HEADS, HEAD_DIM), marker, dtype=torch.float32),
            torch.full((n_tokens, N_HEADS, HEAD_DIM), -marker, dtype=torch.float32),
        ]
        for _ in range(N_LAYERS)
    ]


def _chunk_cache_key(tokens: List[int]) -> str:
    h = hashlib.sha256()
    for t in tokens:
        h.update(t.to_bytes(4, "little", signed=True))
    return f"chunk:{h.hexdigest()}"


def _make_forwards():
    chunk_calls: list = []
    pair_calls: list = []
    instr_calls: list = []

    def run_instr(token_ids, slice_start, slice_end):
        instr_calls.append((tuple(token_ids), slice_start, slice_end))
        return _layers_for(len(token_ids), 100.0)

    def run_chunk(token_ids, slice_start, slice_end):
        chunk_calls.append((tuple(token_ids), slice_start, slice_end))
        return _layers_for(len(token_ids), float(token_ids[0]))

    def run_pair(concat_tokens):
        pair_calls.append(tuple(concat_tokens))
        return _layers_for(len(concat_tokens), 777.0)

    return run_instr, run_chunk, run_pair, instr_calls, chunk_calls, pair_calls


def test_even_retrieval_forms_adjacent_pairs_no_individual_chunks():
    run_instr, run_chunk, run_pair, instr_calls, chunk_calls, pair_calls = _make_forwards()
    # 4 retrieved docs → 2 adjacent pairs, 0 individual chunk forwards.
    instr = [9]
    d0, d1, d2, d3 = [10], [11], [12], [13]
    query = [99]
    doc_chunk_ids = [instr, d0, d1, d2, d3, query]
    retrieval_doc_ids = ["d0", "d1", "d2", "d3"]

    input_ids, kvs, _, stats = assemble_pairs_per_query(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_doc_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        run_pair_forward=run_pair,
        s_start_len=1,
        s_start_1_len=2,
        individual_cache=None,
    )
    assert len(pair_calls) == 2
    assert pair_calls[0] == tuple(d0 + d1)
    assert pair_calls[1] == tuple(d2 + d3)
    # Only instr + query hit run_chunk / run_instr (no individual doc chunks).
    assert len(instr_calls) == 1
    assert len(chunk_calls) == 1  # the query suffix
    assert stats.pair_hits == 2
    assert stats.matched_pairs == [("d0", "d1"), ("d2", "d3")]
    # input_ids preserves original retrieval order (no reordering).
    strip = 2 - 1
    expected = list(instr) + d0[strip:] + d1[strip:] + d2[strip:] + d3[strip:] + query[strip:]
    assert input_ids == expected


def test_odd_retrieval_pairs_prefix_and_forwards_tail_as_individual():
    run_instr, run_chunk, run_pair, _, chunk_calls, pair_calls = _make_forwards()
    d0, d1, d2 = [10], [11], [12]
    doc_chunk_ids = [[9], d0, d1, d2, [99]]
    retrieval_doc_ids = ["d0", "d1", "d2"]

    _, _, _, stats = assemble_pairs_per_query(
        doc_chunk_ids=doc_chunk_ids,
        retrieval_doc_ids=retrieval_doc_ids,
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        run_pair_forward=run_pair,
        s_start_len=1,
        s_start_1_len=2,
        individual_cache=None,
    )
    assert len(pair_calls) == 1
    assert pair_calls[0] == tuple(d0 + d1)
    # query suffix + odd-out d2 both go through run_chunk.
    chunk_toks = [c[0] for c in chunk_calls]
    assert tuple(d2) in chunk_toks
    assert tuple([99]) in chunk_toks
    assert stats.pair_hits == 1
    assert stats.matched_pairs == [("d0", "d1")]


def test_no_cache_second_query_recomputes_all_pairs():
    run_instr, run_chunk, run_pair, _, _, pair_calls = _make_forwards()
    common = dict(
        instr_cache_key="__instr__",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        run_pair_forward=run_pair,
        s_start_len=1,
        s_start_1_len=2,
        individual_cache=None,
    )
    assemble_pairs_per_query(
        doc_chunk_ids=[[9], [10], [11], [99]],
        retrieval_doc_ids=["d0", "d1"],
        query_cache_key="q:1",
        **common,
    )
    assemble_pairs_per_query(
        doc_chunk_ids=[[9], [10], [11], [99]],
        retrieval_doc_ids=["d0", "d1"],
        query_cache_key="q:2",
        **common,
    )
    # Same docs, same pair → both queries force a fresh pair forward (no cache).
    assert len(pair_calls) == 2


def test_consecutive_same_doc_not_paired():
    """Sanity: the canonical pair key rejects (d, d), so adjacent duplicates
    fall through to individual chunk forwards rather than being paired."""
    run_instr, run_chunk, run_pair, _, chunk_calls, pair_calls = _make_forwards()
    _, _, _, stats = assemble_pairs_per_query(
        doc_chunk_ids=[[9], [10], [10], [99]],
        retrieval_doc_ids=["d0", "d0"],
        instr_cache_key="__instr__",
        query_cache_key="q:1",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        run_pair_forward=run_pair,
        s_start_len=1,
        s_start_1_len=2,
        individual_cache=None,
    )
    assert len(pair_calls) == 0
    assert stats.pair_hits == 0
    # Both d0 occurrences go through run_chunk individually (+ query suffix = 3).
    assert len(chunk_calls) == 3


def test_optional_individual_cache_saves_instr_on_reuse():
    cache = FIFOChunkKVCache(max_entries=16)
    run_instr, run_chunk, run_pair, instr_calls, _, _ = _make_forwards()
    common = dict(
        instr_cache_key="__instr__",
        chunk_cache_key_fn=_chunk_cache_key,
        run_instr_forward=run_instr,
        run_chunk_forward=run_chunk,
        run_pair_forward=run_pair,
        s_start_len=1,
        s_start_1_len=2,
        individual_cache=cache,
    )
    assemble_pairs_per_query(
        doc_chunk_ids=[[9], [10], [11], [99]],
        retrieval_doc_ids=["d0", "d1"],
        query_cache_key="q:1",
        **common,
    )
    assemble_pairs_per_query(
        doc_chunk_ids=[[9], [12], [13], [98]],
        retrieval_doc_ids=["d2", "d3"],
        query_cache_key="q:2",
        **common,
    )
    # Instruction prefix computed once, then hit on query 2.
    assert len(instr_calls) == 1
