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
