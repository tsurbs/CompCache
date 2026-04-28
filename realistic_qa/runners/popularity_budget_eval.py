from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SQ_RUNNERS = str(_REPO_ROOT / "standard_qa" / "runners")
if _SQ_RUNNERS not in sys.path:
    sys.path.insert(0, _SQ_RUNNERS)

from utils import CoRetrievalTracker, get_doc_ids, load_dataset
from co_retrieval_logger import CoRetrievalLogger
from composition_cache import CompositionCache
from kv_fifo_cache import FIFOChunkKVCache
from pair_kv_store import (
    FullJointPairStore,
    PairKVStore,
    SparseDeltaPairStore,
)
from promotion_worker import PromotionWorker

METHOD_FULL = "full"
METHOD_SINGLE = "single"
METHOD_LFU_FULL = "lfu_full"
METHOD_LFU_DELTA = "lfu_delta"
METHODS = (METHOD_FULL, METHOD_SINGLE, METHOD_LFU_FULL, METHOD_LFU_DELTA)

METHOD_LABELS = {
    METHOD_FULL: "Full Prefill",
    METHOD_SINGLE: "CompCache (single-chunk)",
    METHOD_LFU_FULL: "CompCache + pairs (Full, LFU)",
    METHOD_LFU_DELTA: "CompCache + pairs (Δ-sparse, LFU, equal bytes)",
}
METHOD_COLORS = {
    METHOD_FULL: "#2ca02c",
    METHOD_SINGLE: "#ff7f0e",
    METHOD_LFU_FULL: "#1f77b4",
    METHOD_LFU_DELTA: "#9467bd",
}

@dataclass
class MethodState:
    name: str
    ttft: List[float] = field(default_factory=list)
    collect_seconds: List[float] = field(default_factory=list)
    metric: List[float] = field(default_factory=list)
    pair_hits: List[int] = field(default_factory=list)
    pair_misses: List[int] = field(default_factory=list)
    memory_log: List[Tuple[int, int, int]] = field(default_factory=list)
    
    fifo: Optional[FIFOChunkKVCache] = None
    pair_store: Optional[PairKVStore] = None
    logger: Optional[CoRetrievalLogger] = None
    worker: Optional[PromotionWorker] = None
    composition: Optional[CompositionCache] = None

def _chunk_cache_key(chunk_index: int, token_ids: List[int]) -> str:
    if chunk_index == 0:
        return "__instr_prefix__"
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return f"ctx:{h.hexdigest()}"

def run_popularity_budget_eval(
    dataset_path: str,
    prefix_prompt: str,
    prompt_builder,
    metric_fn,
    metric_name: str,
    *,
    inst_tokens=None,
    s_end=None,
    suffix_is_query_len: bool = True,
    max_ctx_len=None,
    max_tokens: int = 32,
    recomp_ratio=None,
    pair_recomp_ratio=None,
    fast_attention=None,
    extra_metadata=None,
    post_process=None,
    clear_hack_kv: bool = False,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization: float = 0.45,
    max_model_len: int | None = None,
    num_layers: int = 32,
    stream_seed: int = 34,
    skip_first: int = 0,
    fifo_max_chunks: int = 10_000,
    cap_full: int = 256,
    delta_top_k_ratio: float = 0.10,
    cap_delta: Optional[int] = None,
    promotion_threshold: int = 10,
    shuffle_dataset: bool = True,
    standard_qa: bool = False,
):
    import numpy as np
    import torch
    from itertools import chain
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    full_dataset = load_dataset(dataset_path)
    if shuffle_dataset:
        rng = __import__("random").Random(stream_seed)
        full_dataset = list(full_dataset)
        rng.shuffle(full_dataset)
    eval_dataset = full_dataset[skip_first:] if skip_first else full_dataset

    if cap_delta is None:
        cap_delta = max(cap_full, int(round(cap_full / max(delta_top_k_ratio, 1e-6))))
    print(
        f"[budget] cap_full={cap_full} cap_delta={cap_delta} "
        f"(ratio={delta_top_k_ratio:g}, budget parity ~ cap_full × full_bytes ≈ cap_delta × delta_bytes)"
    )

    llm_kwargs: dict = {
        "model": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm.set_tokenizer(tokenizer)

    if s_end is None:
        s_end = []
    if extra_metadata is None:
        extra_metadata = {}

    prefix_ids = tokenizer.encode(prefix_prompt)[1:]
    s_start_full = list(inst_tokens) + prefix_ids if inst_tokens else prefix_ids
    s_start_len = len(s_start_full) + 1

    s_start: list = []
    s_start_1_len = len(s_start) + 1

    model_ref = (
        llm.llm_engine.model_executor.driver_worker
        .model_runner.model.model
    )
    layers = model_ref.layers
    cache_fuse_metadata = model_ref.cache_fuse_metadata
    gpu_lock = threading.Lock()

    sampling_params_collect = SamplingParams(temperature=0, max_tokens=1)

    def _extract_layers(slice_start: int, slice_end: int) -> list:
        out: list = []
        for j in range(num_layers):
            past_key_values = layers[j].self_attn.hack_kv
            temp_k = past_key_values[0][slice_start:slice_end].clone()
            temp_v = past_key_values[1][slice_start:slice_end].clone()
            out.append([temp_k, temp_v])
            if clear_hack_kv:
                layers[j].self_attn.hack_kv = None
        return out

    def _run_forward(token_ids, slice_start, slice_end):
        prompts = [tokenizer.decode(token_ids)]
        llm.generate(prompts, sampling_params_collect)
        return _extract_layers(slice_start, slice_end)

    def _run_pair_forward(concat_tokens):
        prev_collect = cache_fuse_metadata.get("collect", False)
        prev_check = cache_fuse_metadata.get("check", False)
        cache_fuse_metadata["collect"] = True
        cache_fuse_metadata["check"] = False
        try:
            prompts = [tokenizer.decode(concat_tokens)]
            llm.generate(prompts, sampling_params_collect)
            return _extract_layers(s_start_1_len, len(concat_tokens) + 1)
        finally:
            cache_fuse_metadata["collect"] = prev_collect
            cache_fuse_metadata["check"] = prev_check

    
    states: dict[str, MethodState] = {m: MethodState(name=m) for m in METHODS}

    
    single_fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=True)
    single_logger = CoRetrievalLogger(promotion_threshold=promotion_threshold)
    null_pair_store = FullJointPairStore(1, store_on_cpu=True)
    states[METHOD_SINGLE].fifo = single_fifo
    states[METHOD_SINGLE].logger = single_logger
    states[METHOD_SINGLE].pair_store = null_pair_store
    states[METHOD_SINGLE].composition = CompositionCache(
        individual_cache=single_fifo,
        pair_store=null_pair_store,
        logger=single_logger,
        promotion_worker=None,
        promote_sync=False,
    )

    
    
    
    def _mk_lfu(method: str, store_builder):
        fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=True)
        logger = CoRetrievalLogger(promotion_threshold=promotion_threshold)

        def priority_fn(a: str, b: str, _lg=logger) -> float:
            return float(_lg.count(a, b))

        store = store_builder(priority_fn)
        worker = PromotionWorker(store, _run_pair_forward, gpu_lock)
        worker.start()
        comp = CompositionCache(
            individual_cache=fifo,
            pair_store=store,
            logger=logger,
            promotion_worker=worker,
            promote_sync=False,
        )
        st = states[method]
        st.fifo = fifo
        st.pair_store = store
        st.logger = logger
        st.worker = worker
        st.composition = comp

    _mk_lfu(
        METHOD_LFU_FULL,
        lambda pfn: FullJointPairStore(
            cap_full,
            store_on_cpu=True,
            evict_policy="lfu",
            priority_fn=pfn,
        ),
    )
    _mk_lfu(
        METHOD_LFU_DELTA,
        lambda pfn: SparseDeltaPairStore(
            cap_delta,
            top_k_ratio=delta_top_k_ratio,
            store_on_cpu=True,
            evict_policy="lfu",
            priority_fn=pfn,
        ),
    )

    tracker = CoRetrievalTracker()

    def _run_cached_prefill(input_prompt: str, last_len: int, recomp_override: Optional[float] = None):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["suffix_len"] = last_len
        eff = recomp_override if recomp_override is not None else recomp_ratio
        if eff is not None:
            cache_fuse_metadata["recomp_ratio"] = eff
        if fast_attention is not None:
            cache_fuse_metadata["fast_attention"] = fast_attention
        out = llm.generate([input_prompt], sp)
        ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
        text = out[0].outputs[0].text
        if post_process:
            text = post_process(text)
        return ttft, text

    def _run_full_prefill(input_prompt: str):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = False
        cache_fuse_metadata["collect"] = False
        out = llm.generate([input_prompt], sp)
        ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
        text = out[0].outputs[0].text
        if post_process:
            text = post_process(text)
        return ttft, text

    try:
        for sample_idx, ex in enumerate(eval_dataset):
            answers = ex["answers"]
            doc_prompts, q_prompt = prompt_builder(ex)
            doc_ids = get_doc_ids(ex)
            tracker.record(doc_ids)

            doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]
            q_ids = tokenizer.encode(q_prompt)[1:]

            if max_ctx_len is not None:
                while len(list(chain.from_iterable(doc_chunk_ids))) > max_ctx_len:
                    del_idx = int(len(doc_chunk_ids) / 2)
                    del doc_chunk_ids[del_idx]
                    if del_idx < len(doc_ids):
                        del doc_ids[del_idx]
                if len(doc_chunk_ids) == 0:
                    continue
            doc_ids = list(doc_ids[: len(doc_chunk_ids)])

            doc_chunk_ids = [s_start + cids for cids in doc_chunk_ids]
            doc_chunk_ids = [s_start_full] + doc_chunk_ids
            doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]
            last_len = len(q_ids + s_end) if suffix_is_query_len else len([q_ids + s_end])

            full_input_ids = list(doc_chunk_ids[0])
            for c in doc_chunk_ids[1:]:
                full_input_ids.extend(c[s_start_1_len - 1:])
            full_input_prompt = tokenizer.decode(full_input_ids)

            
            with gpu_lock:
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t_ttft, text = _run_full_prefill(full_input_prompt)
            st = states[METHOD_FULL]
            st.ttft.append(t_ttft)
            st.collect_seconds.append(0.0)
            st.pair_hits.append(0)
            st.pair_misses.append(0)
            st.memory_log.append((sample_idx, 0, 0))
            st.metric.append(max(metric_fn(text, a, tokenizer) for a in answers))

            
            for method, recomp_override in (
                (METHOD_SINGLE, None),
                (METHOD_LFU_FULL, pair_recomp_ratio),
                (METHOD_LFU_DELTA, pair_recomp_ratio),
            ):
                st = states[method]
                with gpu_lock:
                    cache_fuse_metadata["collect"] = True
                    cache_fuse_metadata["check"] = False
                    for k, v in extra_metadata.items():
                        cache_fuse_metadata[k] = v
                    t0 = time.perf_counter()
                    assert st.composition is not None
                    input_ids, kvs, _, qstats = st.composition.assemble(
                        doc_chunk_ids=doc_chunk_ids,
                        retrieval_doc_ids=doc_ids,
                        instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                        query_cache_key=f"q:{method}:{sample_idx}",
                        chunk_cache_key_fn=lambda t: _chunk_cache_key(1, t),
                        run_instr_forward=_run_forward,
                        run_chunk_forward=_run_forward,
                        s_start_len=s_start_len,
                        s_start_1_len=s_start_1_len,
                        disable_pairs=(method == METHOD_SINGLE),
                    )
                    collect_s = time.perf_counter() - t0
                    model_ref.old_kvs = kvs
                    input_prompt = tokenizer.decode(input_ids)
                    ttft, text = _run_cached_prefill(input_prompt, last_len, recomp_override)
                st.ttft.append(ttft)
                st.collect_seconds.append(collect_s)
                st.pair_hits.append(qstats.pair_hits)
                st.pair_misses.append(qstats.pair_misses)
                st.metric.append(max(metric_fn(text, a, tokenizer) for a in answers))
                assert st.pair_store is not None
                st.memory_log.append((
                    sample_idx,
                    st.pair_store.bytes_used(),
                    st.pair_store.stats().get("entries", 0),
                ))

            if sample_idx % 10 == 0 or sample_idx == len(eval_dataset) - 1:
                tfm = states[METHOD_LFU_FULL].memory_log[-1][1] / (1024 ** 2)
                tdm = states[METHOD_LFU_DELTA].memory_log[-1][1] / (1024 ** 2)
                print(
                    f"[budget idx={sample_idx}] "
                    f"full={states[METHOD_FULL].ttft[-1]:.3f}s  "
                    f"single={states[METHOD_SINGLE].ttft[-1]:.3f}s  "
                    f"lfu_full={states[METHOD_LFU_FULL].ttft[-1]:.3f}s ({tfm:.1f}MB)  "
                    f"lfu_delta={states[METHOD_LFU_DELTA].ttft[-1]:.3f}s ({tdm:.1f}MB)",
                    flush=True,
                )
    finally:
        for st in states.values():
            if st.worker is not None and st.worker.is_alive():
                st.worker.stop(drain=False)
                st.worker.join(timeout=5.0)

    
    n = len(states[METHOD_FULL].ttft)

    summary = {}
    for m in METHODS:
        st = states[m]
        arr = np.asarray(st.metric, dtype=float) if st.metric else np.asarray([])
        ttft_arr = np.asarray(st.ttft, dtype=float) if st.ttft else np.asarray([])
        peak_mb = (max((e[1] for e in st.memory_log), default=0)) / (1024.0 ** 2)
        summary[m] = {
            "label": METHOD_LABELS[m],
            f"mean_{metric_name}": float(arr.mean()) if arr.size else 0.0,
            "mean_ttft_seconds": float(ttft_arr.mean()) if ttft_arr.size else 0.0,
            "peak_mb": peak_mb,
            "pair_hits_total": int(sum(st.pair_hits)),
            "pair_misses_total": int(sum(st.pair_misses)),
            "pair_store_stats": st.pair_store.stats() if st.pair_store is not None else None,
            "logger_summary": st.logger.summary() if st.logger is not None else None,
        }

    base = Path(dataset_path).resolve()
    scores_path = base.with_name(f"{base.stem}_budget_scores.json")

    payload = {
        "dataset": str(base),
        "n_queries": n,
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "cap_full": cap_full,
        "cap_delta": cap_delta,
        "delta_top_k_ratio": delta_top_k_ratio,
        "promotion_threshold": promotion_threshold,
        "metric_name": metric_name,
        "per_method_summary": summary,
        "per_query": {
            m: {
                "ttft": [float(x) for x in states[m].ttft],
                "collect_seconds": [float(x) for x in states[m].collect_seconds],
                metric_name: [float(x) for x in states[m].metric],
                "pair_hits": states[m].pair_hits,
                "pair_misses": states[m].pair_misses,
                "memory_log": [[i, int(b), int(e)] for (i, b, e) in states[m].memory_log],
            }
            for m in METHODS
        },
        "coretrieval": tracker.summary(),
        "eval": "popularity_budget",
    }
    with open(scores_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[budget] scores saved to {scores_path}")

    return {
        "n_queries": n,
        "per_method_summary": summary,
        "artifacts": {
            "scores_json": str(scores_path),
            "main_png": None,
            "ttft_hist_png": None,
        },
    }
