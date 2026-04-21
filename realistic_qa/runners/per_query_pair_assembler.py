"""Per-query no-cache pair assembler used by the standard QA 3-way runner.

For each query, the retrieved doc list ``[d0, d1, d2, d3, ...]`` is grouped into
adjacent disjoint pairs ``[(d0, d1), (d2, d3), ...]``. For each pair we run a
*joint* prefill forward (``[d_i + d_{i+1}]``) and snapshot the resulting per-pair
KV; we then concatenate those pair-KVs (plus instruction prefix and query
suffix KVs) and hand them to vLLM via ``model_ref.old_kvs`` exactly as the
existing CompCache assemble does. No caching across queries.

Operationally this simulates a 100% pair-store hit rate without needing
cross-query reuse: every pair we will look at *is* freshly computed and used
on the same query.

The function returns the same ``(input_ids, kvs, reordered_chunks, stats)``
tuple as :meth:`composition_cache.CompositionCache.assemble` so the caller in
``three_way_eval`` can swap implementations without touching the model code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from composition_cache import CollectionForward, PairForward, QueryStats
from kv_fifo_cache import FIFOChunkKVCache, StackedLayers


def _append_layers(acc: StackedLayers, added: StackedLayers) -> None:
    if not acc:
        for k, v in added:
            acc.append([k, v])
        return
    for j, (k, v) in enumerate(added):
        acc[j][0] = torch.cat((acc[j][0], k), dim=0)
        acc[j][1] = torch.cat((acc[j][1], v), dim=0)


def assemble_pairs_per_query(
    *,
    doc_chunk_ids: Sequence[List[int]],
    retrieval_doc_ids: Sequence[str],
    instr_cache_key: str,
    query_cache_key: str,
    chunk_cache_key_fn: Callable[[List[int]], str],
    run_instr_forward: CollectionForward,
    run_chunk_forward: CollectionForward,
    run_pair_forward: PairForward,
    s_start_len: int,
    s_start_1_len: int,
    individual_cache: Optional[FIFOChunkKVCache] = None,
) -> Tuple[List[int], StackedLayers, List[List[int]], QueryStats]:
    """Per-query joint-pair assembler with no cross-query pair caching.

    Pairing strategy: greedy adjacent — ``(d_0, d_1), (d_2, d_3), ...``. If the
    retrieval list has odd length, the last doc is forwarded as a singleton
    chunk via ``run_chunk_forward``.

    ``individual_cache`` (optional) is consulted only for the instruction
    prefix and query suffix (the common-prefix part most workloads will hit on
    repeated queries); the per-pair joint KVs are NEVER stored there. The
    purpose is just to avoid re-encoding the instruction every query when the
    realistic runner hands us a stateful FIFO. Pass ``None`` for "no caching
    at all" semantics.

    ``stats.pair_hits`` is incremented for every pair we actually used (so the
    +pairs path reports a 100% hit rate on every query), while
    ``stats.individual_*`` track the prefix/suffix/odd-out chunks.
    """
    n_docs = len(retrieval_doc_ids)
    assert len(doc_chunk_ids) == n_docs + 2, (
        f"expected {n_docs + 2} chunks (instr + {n_docs} docs + query), "
        f"got {len(doc_chunk_ids)}"
    )

    stats = QueryStats()
    chunk_past_key_values: StackedLayers = []

    instr_layers = _get_or_collect_individual(
        cache=individual_cache,
        key=instr_cache_key,
        token_ids=list(doc_chunk_ids[0]),
        run_forward=run_instr_forward,
        slice_start=0,
        slice_end=s_start_len,
        stats=stats,
    )
    for layer_k, layer_v in instr_layers:
        chunk_past_key_values.append([layer_k, layer_v])

    matched_pairs: List[Tuple[str, str]] = []

    i = 0
    while i < n_docs:
        if i + 1 < n_docs and retrieval_doc_ids[i] != retrieval_doc_ids[i + 1]:
            tokens_a = list(doc_chunk_ids[1 + i])
            tokens_b = list(doc_chunk_ids[1 + i + 1])
            concat = tokens_a + tokens_b
            joint = run_pair_forward(concat)
            _append_layers(chunk_past_key_values, joint)
            stats.pair_hits += 1
            matched_pairs.append((retrieval_doc_ids[i], retrieval_doc_ids[i + 1]))
            i += 2
        else:
            tokens = list(doc_chunk_ids[1 + i])
            layer_kvs = _get_or_collect_individual(
                cache=None,
                key=chunk_cache_key_fn(tokens),
                token_ids=tokens,
                run_forward=run_chunk_forward,
                slice_start=s_start_1_len,
                slice_end=len(tokens) + 1,
                stats=stats,
            )
            _append_layers(chunk_past_key_values, layer_kvs)
            i += 1

    query_tokens = list(doc_chunk_ids[-1])
    query_layers = _get_or_collect_individual(
        cache=individual_cache,
        key=query_cache_key,
        token_ids=query_tokens,
        run_forward=run_chunk_forward,
        slice_start=s_start_1_len,
        slice_end=len(query_tokens) + 1,
        stats=stats,
    )
    _append_layers(chunk_past_key_values, query_layers)

    reordered_chunks: List[List[int]] = [list(doc_chunk_ids[0])]
    for k in range(n_docs):
        reordered_chunks.append(list(doc_chunk_ids[1 + k]))
    reordered_chunks.append(list(doc_chunk_ids[-1]))

    strip = s_start_1_len - 1
    input_ids: List[int] = list(reordered_chunks[0])
    for chunk in reordered_chunks[1:]:
        input_ids.extend(chunk[strip:])

    stats.matched_pairs = matched_pairs
    return input_ids, chunk_past_key_values, reordered_chunks, stats


def _get_or_collect_individual(
    *,
    cache: Optional[FIFOChunkKVCache],
    key: str,
    token_ids: List[int],
    run_forward: CollectionForward,
    slice_start: int,
    slice_end: int,
    stats: QueryStats,
) -> StackedLayers:
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            stats.individual_hits += 1
            return cached
    stats.individual_misses += 1
    layers = run_forward(token_ids, slice_start, slice_end)
    if cache is not None:
        cache.put(key, layers)
    return layers
