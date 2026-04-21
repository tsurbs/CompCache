"""Composition-aware KV cache orchestrator (Proposal §3.1).

Sits between the retriever and the CacheBlend fusion step. For each query:

1. Match pairs of retrieved docs against the promoted-pair store (maximum
   disjoint coverage).
2. Reorder retrieved chunks so matched pair members are adjacent in canonical
   order (non-pair chunks keep their original position).
3. Assemble the per-layer KV by, in order:
   - joint pair cache hits  → one KV block spanning both chunks
   - individual chunk hits  → FIFO cache
   - misses                 → fresh collection forward on the engine
4. Log co-retrievals and enqueue promotion jobs for pairs that just crossed
   the threshold.
5. Return the re-assembled ``input_ids`` (token IDs in the new order),
   the concatenated ``old_kvs``, and per-query hit statistics.

Position 0 of ``doc_chunk_ids`` is the instruction prefix and position ``-1``
is the query suffix; both are always treated individually (prefix uses the
shared ``__instr_prefix__`` key, query suffix is unique per request).
"""
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


# run_chunk_forward(token_ids, slice_start, slice_end) → per-layer [K,V]
CollectionForward = Callable[[List[int], int, int], StackedLayers]

# run_pair_forward(concat_tokens) → per-layer [K,V] for the joint pair prefill,
# starting from the s_start_1 boundary (matches three_way_eval._run_pair_forward).
PairForward = Callable[[List[int]], StackedLayers]


class CompositionCache:
    """Owns the individual FIFO, pair store, promotion worker, and logger."""

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

    # ---- main entry -------------------------------------------------------

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
        """Produce input_ids, old_kvs, reordered_chunks, and stats for one query.

        ``doc_chunk_ids`` follows the existing convention:
            [0]    instruction prefix (s_start_full)
            [1:-1] retrieved document chunks (len == len(retrieval_doc_ids))
            [-1]   query suffix

        ``disable_pairs=True`` skips pair matching, pair-store lookups, the
        co-retrieval logger, and promotion enqueue, so this orchestrator runs
        as a pure "chunk-FIFO + selective recompute" path (CacheBlend Method 2
        in the three-way comparison).  Used by the three-way runner; existing
        callers default to the full composition-aware behavior.

        ``treat_all_pairs_as_cached=True`` (mutually exclusive with
        ``disable_pairs``) bypasses the CoRetrievalLogger entirely: the matcher
        is invoked with ``is_cached=lambda a,b: True`` so every adjacent pair
        in the retrieval is matched, and on a pair-store miss the joint KV is
        computed *synchronously* via ``run_pair_forward(concat_tokens)`` and
        inserted into the (FIFO) pair store before being consumed.  No
        promotion threshold; the pair store is the cache.
        """
        if disable_pairs and treat_all_pairs_as_cached:
            raise ValueError(
                "disable_pairs and treat_all_pairs_as_cached are mutually exclusive"
            )
        if treat_all_pairs_as_cached and run_pair_forward is None:
            raise ValueError(
                "treat_all_pairs_as_cached=True requires run_pair_forward"
            )

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
            # Match greedily across all pairs (predicate always True), with
            # frequency = 0 so the matcher's tiebreak just falls back to
            # coverage. We sync-compute any missing pair KVs below.
            matches = find_best_matching(
                retrieval_doc_ids,
                is_cached=lambda a, b: True,
            )
        else:
            # 1. Log this query's co-retrievals, get promotion candidates for later enqueue.
            newly_ready = self.logger.record(retrieval_doc_ids)

            # 2. Pair matching over retrieval positions 0..n_docs-1.
            matches = find_best_matching(
                retrieval_doc_ids,
                is_cached=self.pair_store.contains,
                pair_frequency=lambda a, b: self.logger.count(a, b),
            )
        stats.matched_pairs = [m.pair_key for m in matches]

        # When treat_all_pairs_as_cached: ensure every matched pair is in the
        # store before we walk the chunk list. On a miss, sync-compute via
        # run_pair_forward (which jointly prefills [s_start + a + b] and snapshots
        # the per-pair KV) and put() it into the pair store (FIFO eviction).
        if treat_all_pairs_as_cached:
            for m in matches:
                doc_a, doc_b = m.pair_key
                if self.pair_store.contains(doc_a, doc_b):
                    continue
                tokens_a = list(doc_chunk_ids[1 + m.positions[0]])
                tokens_b = list(doc_chunk_ids[1 + m.positions[1]])
                # Mirrors PromotionWorker._process: feed the raw concat to the
                # joint forward; ``run_pair_forward`` is responsible for the
                # ``[s_start_1_len, len(concat)+1]`` slice that produces a KV of
                # shape [len(tokens_a) + len(tokens_b), ...].
                concat = tokens_a + tokens_b
                joint = run_pair_forward(concat)  # type: ignore[misc]
                self.pair_store.put(doc_a, doc_b, joint)

        # 3. Reorder retrieval positions so pair members are adjacent canonically.
        reorder = apply_reordering(retrieval_doc_ids, matches)
        # Map each pair's retrieval-position tuple → PairMatch for quick lookup below.
        pair_starts: dict[int, PairMatch] = {m.positions[0]: m for m in matches}

        # 4. Walk the reordered chunk list and collect per-layer KVs.
        chunk_past_key_values: StackedLayers = []

        # ---- instruction prefix ----
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

        # ---- retrieved docs, in the reordered order ----
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

            # Single chunk (either unmatched or pair cache miss). If the pair
            # was reordered adjacent but its cache missed, we need to fall
            # back to per-chunk assembly for BOTH positions — handled
            # naturally by the while-loop: this iteration processes pos,
            # the next iteration processes match.positions[1].
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

        # ---- query suffix (always individual; usually a miss) ----
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

        # 5. Build reordered chunk list (for input_ids assembly by the caller).
        reordered_chunks: List[List[int]] = [list(doc_chunk_ids[0])]
        for pos in reorder:
            reordered_chunks.append(list(doc_chunk_ids[1 + pos]))
        reordered_chunks.append(list(doc_chunk_ids[-1]))

        # 6. Build concatenated input_ids using the existing scheme (strip s_start_1_len-1
        #    from every non-first chunk to avoid repeating any leading s_start tokens).
        strip = s_start_1_len - 1
        input_ids: List[int] = list(reordered_chunks[0])
        for chunk in reordered_chunks[1:]:
            input_ids.extend(chunk[strip:])

        # 7. Enqueue promotion jobs for pairs that just hit the threshold on this query.
        # (No-op for disable_pairs and treat_all_pairs_as_cached: ``newly_ready`` is empty.)
        needs_individuals = getattr(self.pair_store, "needs_individuals", False)
        if newly_ready and self.worker is not None:
            for doc_a, doc_b in newly_ready:
                if self.pair_store.contains(doc_a, doc_b):
                    continue
                tokens_a, tokens_b = self._tokens_for(
                    doc_a, doc_b, retrieval_doc_ids, doc_chunk_ids
                )
                if tokens_a is None or tokens_b is None:
                    # The pair was flagged on some past query but those tokens aren't
                    # in the current retrieval; we rely on a future co-retrieval where
                    # they both appear. Skip rather than snapshotting now.
                    continue
                ind_a: Optional[StackedLayers] = None
                ind_b: Optional[StackedLayers] = None
                if needs_individuals:
                    # Delta stores need the individuals to compute Δ at put-time.
                    # We just assembled this query, so both should be warm in
                    # the FIFO; if not, skip the promotion and wait for a
                    # future co-retrieval.
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

    # ---- helpers ----------------------------------------------------------

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
        """Attempt pair cache hit; return True if we appended the joint KV.

        For delta-backed stores (``pair_store.needs_individuals``), we fetch
        the individuals from the FIFO and pass them in so the store can
        reconstruct the joint tensor. On a reconstruction miss we fall
        through to individual assembly.
        """
        doc_a, doc_b = match.pair_key
        if getattr(self.pair_store, "needs_individuals", False):
            ind_a = self.individual_cache.get(chunk_cache_key_fn(tokens_a))
            ind_b = self.individual_cache.get(chunk_cache_key_fn(tokens_b))
            if ind_a is None or ind_b is None:
                # Can't reconstruct without individuals. Fall back to
                # per-chunk assembly (the two single-chunk passes will
                # repopulate the FIFO).
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
        # Sanity: stored joint KV should cover the full concatenation.
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
        # Prefix and all retrieved chunks share the same FIFO.
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
