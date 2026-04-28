from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from co_retrieval_logger import CoRetrievalLogger
from kv_fifo_cache import FIFOChunkKVCache
from pair_kv_store import PairKVStore, StackedLayers
from pair_matcher import PairMatch, apply_reordering, find_best_matching
from promotion_worker import PromotionJob, PromotionWorker

@dataclass
class QueryStats:
    pair_hits: int = 0
    pair_misses: int = 0
    individual_hits: int = 0
    individual_misses: int = 0
    promotions_enqueued: int = 0
    matched_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pair_hits": self.pair_hits,
            "pair_misses": self.pair_misses,
            "individual_hits": self.individual_hits,
            "individual_misses": self.individual_misses,
            "promotions_enqueued": self.promotions_enqueued,
            "matched_pairs": [list(p) for p in self.matched_pairs],
        }

CollectionForward = Callable[[List[int], int, int], StackedLayers]
PairForward = Callable[[List[int]], StackedLayers]

# Assemble one query: FIFO chunk cache + optional pair KVs + co-retrieval logging.
class CompositionCache:
    def __init__(
        self,
        *,
        individual_cache: FIFOChunkKVCache,
        pair_store: PairKVStore,
        logger: CoRetrievalLogger,
        promotion_worker: Optional[PromotionWorker],
        promote_sync: bool = False,
    ) -> None:
        self.individual_cache = individual_cache
        self.pair_store = pair_store
        self.logger = logger
        self.worker = promotion_worker
        self.promote_sync = promote_sync

    def assemble(
        self,
        *,
        doc_chunk_ids: Sequence[List[int]],
        retrieval_doc_ids: Sequence[str],
        instr_cache_key: str,
        query_cache_key: str,
        chunk_cache_key_fn: Callable[[List[int]], str],
        run_instr_forward: CollectionForward,
        run_chunk_forward: CollectionForward,
        s_start_len: int,
        s_start_1_len: int,
        disable_pairs: bool = False,
        treat_all_pairs_as_cached: bool = False,
        run_pair_forward: Optional[PairForward] = None,
    ) -> Tuple[List[int], StackedLayers, List[List[int]], QueryStats]:
        if disable_pairs and treat_all_pairs_as_cached:
            treat_all_pairs_as_cached = False
        if treat_all_pairs_as_cached and run_pair_forward is None:
            treat_all_pairs_as_cached = False

        n_docs = len(retrieval_doc_ids)
        assert len(doc_chunk_ids) == n_docs + 2, (
            f"expected {n_docs + 2} chunks (instr + {n_docs} docs + query), "
            f"got {len(doc_chunk_ids)}"
        )

        stats = QueryStats()

        if disable_pairs:
            newly_ready: List[Tuple[str, str]] = []
            matches: List[PairMatch] = []
        elif treat_all_pairs_as_cached:
            newly_ready = []
            matches = find_best_matching(
                retrieval_doc_ids,
                is_cached=lambda a, b: True,
            )
        else:
            newly_ready = self.logger.record(retrieval_doc_ids)
            matches = find_best_matching(
                retrieval_doc_ids,
                is_cached=self.pair_store.contains,
                pair_frequency=lambda a, b: self.logger.count(a, b),
            )
        stats.matched_pairs = [m.pair_key for m in matches]

        if treat_all_pairs_as_cached:
            for m in matches:
                doc_a, doc_b = m.pair_key
                if self.pair_store.contains(doc_a, doc_b):
                    continue
                tokens_a = list(doc_chunk_ids[1 + m.positions[0]])
                tokens_b = list(doc_chunk_ids[1 + m.positions[1]])
                concat = tokens_a + tokens_b
                joint = run_pair_forward(concat)  # type: ignore[misc]
                self.pair_store.put(doc_a, doc_b, joint)

        reorder = apply_reordering(retrieval_doc_ids, matches)
        pair_starts: dict[int, PairMatch] = {m.positions[0]: m for m in matches}

        chunk_past_key_values: StackedLayers = []

        instr_layers = self._get_or_collect_individual(
            key=instr_cache_key,
            token_ids=doc_chunk_ids[0],
            run_forward=run_instr_forward,
            slice_start=0,
            slice_end=s_start_len,
            stats=stats,
            is_prefix=True,
        )
        for layer_k, layer_v in instr_layers:
            chunk_past_key_values.append([layer_k, layer_v])

        i = 0
        while i < len(reorder):
            pos = reorder[i]
            match = pair_starts.get(pos)
            consumed_pair = False
            if match is not None and i + 1 < len(reorder) and reorder[i + 1] == match.positions[1]:
                consumed_pair = self._try_consume_pair(
                    match=match,
                    tokens_a=doc_chunk_ids[1 + match.positions[0]],
                    tokens_b=doc_chunk_ids[1 + match.positions[1]],
                    chunk_past_key_values=chunk_past_key_values,
                    stats=stats,
                    chunk_cache_key_fn=chunk_cache_key_fn,
                )

            if consumed_pair:
                i += 2
                continue

            tokens = doc_chunk_ids[1 + pos]
            layer_kvs = self._get_or_collect_individual(
                key=chunk_cache_key_fn(tokens),
                token_ids=tokens,
                run_forward=run_chunk_forward,
                slice_start=s_start_1_len,
                slice_end=len(tokens) + 1,
                stats=stats,
                is_prefix=False,
            )
            self._append_layers(chunk_past_key_values, layer_kvs)
            i += 1

        query_tokens = doc_chunk_ids[-1]
        query_layers = self._get_or_collect_individual(
            key=query_cache_key,
            token_ids=query_tokens,
            run_forward=run_chunk_forward,
            slice_start=s_start_1_len,
            slice_end=len(query_tokens) + 1,
            stats=stats,
            is_prefix=False,
        )
        self._append_layers(chunk_past_key_values, query_layers)

        reordered_chunks: List[List[int]] = [list(doc_chunk_ids[0])]
        for pos in reorder:
            reordered_chunks.append(list(doc_chunk_ids[1 + pos]))
        reordered_chunks.append(list(doc_chunk_ids[-1]))

        strip = s_start_1_len - 1
        input_ids: List[int] = list(reordered_chunks[0])
        for chunk in reordered_chunks[1:]:
            input_ids.extend(chunk[strip:])

        needs_individuals = getattr(self.pair_store, "needs_individuals", False)
        if newly_ready and self.worker is not None:
            for doc_a, doc_b in newly_ready:
                if self.pair_store.contains(doc_a, doc_b):
                    continue
                tokens_a, tokens_b = self._tokens_for(
                    doc_a, doc_b, retrieval_doc_ids, doc_chunk_ids
                )
                if tokens_a is None or tokens_b is None:
                    continue
                ind_a: Optional[StackedLayers] = None
                ind_b: Optional[StackedLayers] = None
                if needs_individuals:
                    ind_a = self.individual_cache.get(chunk_cache_key_fn(tokens_a))
                    ind_b = self.individual_cache.get(chunk_cache_key_fn(tokens_b))
                    if ind_a is None or ind_b is None:
                        continue
                job = PromotionJob(
                    doc_a,
                    doc_b,
                    tokens_a,
                    tokens_b,
                    individual_a=ind_a,
                    individual_b=ind_b,
                )
                if self.promote_sync:
                    self.worker.promote_sync(job)
                    stats.promotions_enqueued += 1
                elif self.worker.enqueue(job):
                    stats.promotions_enqueued += 1

        return input_ids, chunk_past_key_values, reordered_chunks, stats

    def _try_consume_pair(
        self,
        *,
        match: PairMatch,
        tokens_a: List[int],
        tokens_b: List[int],
        chunk_past_key_values: StackedLayers,
        stats: QueryStats,
        chunk_cache_key_fn: Callable[[List[int]], str],
    ) -> bool:
        doc_a, doc_b = match.pair_key
        if getattr(self.pair_store, "needs_individuals", False):
            ind_a = self.individual_cache.get(chunk_cache_key_fn(tokens_a))
            ind_b = self.individual_cache.get(chunk_cache_key_fn(tokens_b))
            if ind_a is None or ind_b is None:
                stats.pair_misses += 1
                return False
            joint = self.pair_store.get(
                doc_a, doc_b, individual_a=ind_a, individual_b=ind_b
            )
        else:
            joint = self.pair_store.get(doc_a, doc_b)
        if joint is None:
            stats.pair_misses += 1
            return False
        expected_len = len(tokens_a) + len(tokens_b)
        if joint[0][0].shape[0] != expected_len:
            stats.pair_misses += 1
            return False
        self._append_layers(chunk_past_key_values, joint)
        stats.pair_hits += 1
        return True

    def _get_or_collect_individual(
        self,
        *,
        key: str,
        token_ids: List[int],
        run_forward: CollectionForward,
        slice_start: int,
        slice_end: int,
        stats: QueryStats,
        is_prefix: bool,
    ) -> StackedLayers:
        cached = self.individual_cache.get(key)
        if cached is not None:
            stats.individual_hits += 1
            return cached
        stats.individual_misses += 1
        layers = run_forward(token_ids, slice_start, slice_end)
        self.individual_cache.put(key, layers)
        return layers

    @staticmethod
    def _append_layers(acc: StackedLayers, added: StackedLayers) -> None:
        if not acc:
            for k, v in added:
                acc.append([k, v])
            return
        for j, (k, v) in enumerate(added):
            acc[j][0] = torch.cat((acc[j][0], k), dim=0)
            acc[j][1] = torch.cat((acc[j][1], v), dim=0)

    @staticmethod
    def _tokens_for(
        doc_a: str,
        doc_b: str,
        retrieval_doc_ids: Sequence[str],
        doc_chunk_ids: Sequence[List[int]],
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        tokens_a = tokens_b = None
        for idx, did in enumerate(retrieval_doc_ids):
            if tokens_a is None and did == doc_a:
                tokens_a = list(doc_chunk_ids[1 + idx])
            elif tokens_b is None and did == doc_b:
                tokens_b = list(doc_chunk_ids[1 + idx])
        return tokens_a, tokens_b
