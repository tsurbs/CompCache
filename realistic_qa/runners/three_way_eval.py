from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SQ_RUNNERS = str(_REPO_ROOT / "standard_qa" / "runners")
if _SQ_RUNNERS not in sys.path:
    sys.path.insert(0, _SQ_RUNNERS)

from utils import CoRetrievalTracker, get_doc_ids, load_dataset
from ttft_reporting import save_ttft_histogram, save_ttft_warmup_plot
from co_retrieval_logger import CoRetrievalLogger
from composition_cache import CompositionCache
from kv_fifo_cache import FIFOChunkKVCache
from pair_kv_store import FullJointPairStore
from per_query_pair_assembler import assemble_pairs_per_query

def _three_way_suffix() -> str:
    tag = os.environ.get("THREE_WAY_OUTPUT_TAG", "").strip()
    return f"_3way_{tag}" if tag else "_3way"

def _chunk_cache_key(chunk_index: int, token_ids: List[int]) -> str:
    if chunk_index == 0:
        return "__instr_prefix__"
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return f"ctx:{h.hexdigest()}"

def _save_3way_warmup(
    dataset_path: str,
    ttft_full: list[float],
    ttft_single: list[float],
    ttft_pair: list[float],
    *,
    stream_seed: int,
    skip_first: int,
    metadata: dict,
    roll_window: int,
) -> tuple[str, str | None]:
    base = Path(dataset_path).resolve()
    suffix = _three_way_suffix()
    json_path = base.with_name(f"{base.stem}{suffix}_ttft_warmup.json")
    n = len(ttft_full)
    payload = {
        "query_index": list(range(n)),
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "ttft_single_seconds": [float(x) for x in ttft_single],
        "ttft_pair_seconds": [float(x) for x in ttft_pair],
        "metadata": {
            "dataset": str(base),
            "stream_seed": stream_seed,
            "skip_first": skip_first,
            "roll_window": roll_window,
            **metadata,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"3way TTFT warmup data: {json_path}")
    return str(json_path), None

def _save_3way_hist(
    dataset_path: str,
    ttft_full: list[float],
    ttft_single: list[float],
    ttft_pair: list[float],
    *,
    metadata: dict,
) -> tuple[str, str | None]:
    base = Path(dataset_path).resolve()
    suffix = _three_way_suffix()
    json_path = base.with_name(f"{base.stem}{suffix}_ttft_hist.json")
    n = len(ttft_full)
    if n == 0:
        return "", None
    payload = {
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "ttft_single_seconds": [float(x) for x in ttft_single],
        "ttft_pair_seconds": [float(x) for x in ttft_pair],
        "metadata": {**metadata, "dataset": str(base), "n_queries": n},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"3way TTFT histogram data: {json_path}")
    return str(json_path), None

def run_blend_eval_three_way(
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
    gpu_memory_utilization: float = 0.5,
    max_model_len: int | None = None,
    num_layers: int = 32,
    stream_seed: int = 34,
    skip_first: int = 0,
    fifo_max_chunks: int = 10_000,
    pair_store_capacity: int = 256,
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

    
    
    
    
    cache_store_on_cpu = True
    single_fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=cache_store_on_cpu)
    
    
    
    if standard_qa:
        pair_fifo = None
        pair_store = None
    else:
        pair_fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=cache_store_on_cpu)
        pair_store = FullJointPairStore(
            pair_store_capacity, store_on_cpu=cache_store_on_cpu, fifo=True
        )

    
    
    single_logger = CoRetrievalLogger(promotion_threshold=2)
    null_pair_store = FullJointPairStore(1, store_on_cpu=cache_store_on_cpu)
    tracker = CoRetrievalTracker()

    model_ref = (
        llm.llm_engine.model_executor.driver_worker
        .model_runner.model.model
    )
    layers = model_ref.layers
    cache_fuse_metadata = model_ref.cache_fuse_metadata
    gpu_lock = threading.Lock()

    print(
        f"[3way] recomp_ratio(single)={recomp_ratio!r} "
        f"recomp_ratio(+pairs)={pair_recomp_ratio if pair_recomp_ratio is not None else recomp_ratio!r}"
    )

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

    composition_single = CompositionCache(
        individual_cache=single_fifo,
        pair_store=null_pair_store,
        logger=single_logger,
        promotion_worker=None,
        promote_sync=False,
    )
    
    
    
    composition_pair: CompositionCache | None = None
    if not standard_qa:
        composition_pair = CompositionCache(
            individual_cache=pair_fifo,
            pair_store=pair_store,
            logger=CoRetrievalLogger(promotion_threshold=2),
            promotion_worker=None,
            promote_sync=False,
        )

    def _chunk_key_fn(tokens):
        return _chunk_cache_key(1, tokens)

    ttft_full: list[float] = []
    ttft_single: list[float] = []
    ttft_pair: list[float] = []
    collect_full: list[float] = []
    collect_single: list[float] = []
    collect_pair: list[float] = []
    metric_full: list[float] = []
    metric_single: list[float] = []
    metric_pair: list[float] = []
    single_query_stats: list = []
    pair_query_stats: list = []

    def _run_cached_prefill(
        input_prompt: str,
        last_len: int,
        recomp_override: float | None = None,
    ) -> tuple[float, str]:
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["suffix_len"] = last_len
        effective_recomp = recomp_override if recomp_override is not None else recomp_ratio
        if effective_recomp is not None:
            cache_fuse_metadata["recomp_ratio"] = effective_recomp
        if fast_attention is not None:
            cache_fuse_metadata["fast_attention"] = fast_attention
        out = llm.generate([input_prompt], sp)
        ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
        text = out[0].outputs[0].text
        if post_process:
            text = post_process(text)
        return ttft, text

    def _run_full_prefill(input_prompt: str) -> tuple[float, str]:
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

            
            
            

            
            full_input_ids: list[int] = list(doc_chunk_ids[0])
            for c in doc_chunk_ids[1:]:
                full_input_ids.extend(c[s_start_1_len - 1:])
            full_input_prompt = tokenizer.decode(full_input_ids)

            
            with gpu_lock:
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t_ttft_full, text_full = _run_full_prefill(full_input_prompt)
            ttft_full.append(t_ttft_full)
            collect_full.append(0.0)
            score = max(metric_fn(text_full, ans, tokenizer) for ans in answers)
            metric_full.append(score)
            print(f"[3way idx={sample_idx}] full TTFT={t_ttft_full:.4f}s score={score:.3f} -> {text_full!r}")

            
            with gpu_lock:
                cache_fuse_metadata["collect"] = True
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t0 = time.perf_counter()
                input_ids_s, kvs_s, _, qstats_s = composition_single.assemble(
                    doc_chunk_ids=doc_chunk_ids,
                    retrieval_doc_ids=doc_ids,
                    instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                    query_cache_key=f"q:{sample_idx}",
                    chunk_cache_key_fn=_chunk_key_fn,
                    run_instr_forward=_run_forward,
                    run_chunk_forward=_run_forward,
                    s_start_len=s_start_len,
                    s_start_1_len=s_start_1_len,
                    disable_pairs=True,
                )
                collect_s = time.perf_counter() - t0
                model_ref.old_kvs = kvs_s
                input_prompt_s = tokenizer.decode(input_ids_s)
                t_ttft_s, text_s = _run_cached_prefill(input_prompt_s, last_len)
            ttft_single.append(t_ttft_s)
            collect_single.append(collect_s)
            single_query_stats.append(qstats_s.as_dict())
            score = max(metric_fn(text_s, ans, tokenizer) for ans in answers)
            metric_single.append(score)
            print(
                f"[3way idx={sample_idx}] single TTFT={t_ttft_s:.4f}s "
                f"collect={collect_s:.4f}s score={score:.3f} -> {text_s!r}"
            )

            
            with gpu_lock:
                cache_fuse_metadata["collect"] = True
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t0 = time.perf_counter()
                if standard_qa:
                    
                    
                    input_ids_p, kvs_p, _, qstats_p = assemble_pairs_per_query(
                        doc_chunk_ids=doc_chunk_ids,
                        retrieval_doc_ids=doc_ids,
                        instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                        query_cache_key=f"q:{sample_idx}",
                        chunk_cache_key_fn=_chunk_key_fn,
                        run_instr_forward=_run_forward,
                        run_chunk_forward=_run_forward,
                        run_pair_forward=_run_pair_forward,
                        s_start_len=s_start_len,
                        s_start_1_len=s_start_1_len,
                        individual_cache=None,
                    )
                else:
                    assert composition_pair is not None
                    input_ids_p, kvs_p, _, qstats_p = composition_pair.assemble(
                        doc_chunk_ids=doc_chunk_ids,
                        retrieval_doc_ids=doc_ids,
                        instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                        query_cache_key=f"q:{sample_idx}",
                        chunk_cache_key_fn=_chunk_key_fn,
                        run_instr_forward=_run_forward,
                        run_chunk_forward=_run_forward,
                        s_start_len=s_start_len,
                        s_start_1_len=s_start_1_len,
                        treat_all_pairs_as_cached=True,
                        run_pair_forward=_run_pair_forward,
                    )
                collect_p = time.perf_counter() - t0
                model_ref.old_kvs = kvs_p
                input_prompt_p = tokenizer.decode(input_ids_p)
                t_ttft_p, text_p = _run_cached_prefill(
                    input_prompt_p, last_len, recomp_override=pair_recomp_ratio
                )
            ttft_pair.append(t_ttft_p)
            collect_pair.append(collect_p)
            pair_query_stats.append(qstats_p.as_dict())
            score = max(metric_fn(text_p, ans, tokenizer) for ans in answers)
            metric_pair.append(score)
            print(
                f"[3way idx={sample_idx}] +pairs TTFT={t_ttft_p:.4f}s "
                f"collect={collect_p:.4f}s score={score:.3f} -> {text_p!r}"
            )
            print("------------")
    finally:
        pass

    
    n = len(ttft_full)
    total_full = [t + c for t, c in zip(ttft_full, collect_full)]
    total_single = [t + c for t, c in zip(ttft_single, collect_single)]
    total_pair = [t + c for t, c in zip(ttft_pair, collect_pair)]
    print("\n=============== Result Summary (3-way) =====================")
    print(f"n_queries: {n}")
    if n > 0:
        print(f"TTFT  full   = {np.mean(ttft_full):.4f}  single = {np.mean(ttft_single):.4f}  pair = {np.mean(ttft_pair):.4f}")
        print(f"Total full   = {np.mean(total_full):.4f}  single = {np.mean(total_single):.4f}  pair = {np.mean(total_pair):.4f}")
        print(f"{metric_name} full = {np.mean(metric_full):.4f}  single = {np.mean(metric_single):.4f}  pair = {np.mean(metric_pair):.4f}")
    print(f"single FIFO: {single_fifo.stats()}")
    if standard_qa:
        print("pair FIFO:   (n/a - standard_qa uses per-query no-cache pair compute)")
        print("pair store:  (n/a - standard_qa uses per-query no-cache pair compute)")
    else:
        assert pair_fifo is not None and pair_store is not None
        print(f"pair FIFO:   {pair_fifo.stats()}")
        print(f"pair store:  {pair_store.stats()}")

    single_fifo_stat = single_fifo.stats()
    pair_fifo_stat = pair_fifo.stats() if pair_fifo is not None else None
    pair_store_stat = pair_store.stats() if pair_store is not None else None
    coret_stats = tracker.summary()
    coret_stats.update({
        "single_fifo_kv": single_fifo_stat,
        "pair_fifo_kv": pair_fifo_stat,
        "pair_store": pair_store_stat,
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "pair_mode": (
            "standard_per_query_no_cache" if standard_qa
            else "realistic_fifo_treat_all_cached"
        ),
        "recomp_ratio": recomp_ratio,
        "pair_recomp_ratio": pair_recomp_ratio,
    })

    suffix = _three_way_suffix()
    coret_path = dataset_path.replace(".json", f"{suffix}_coretrieval.json")
    with open(coret_path, "w") as f:
        json.dump(coret_stats, f, indent=2, default=str)
    print(f"\n3-way co-retrieval stats saved to {coret_path}")

    scores_path = Path(dataset_path).resolve()
    scores_path = scores_path.with_name(f"{scores_path.stem}{suffix}_scores.json")
    scores_payload = {
        "metric_name": metric_name,
        f"{metric_name}_full":   [float(x) for x in metric_full],
        f"{metric_name}_single": [float(x) for x in metric_single],
        f"{metric_name}_pair":   [float(x) for x in metric_pair],
        f"mean_{metric_name}_full":   float(np.mean(metric_full))   if n else 0.0,
        f"mean_{metric_name}_single": float(np.mean(metric_single)) if n else 0.0,
        f"mean_{metric_name}_pair":   float(np.mean(metric_pair))   if n else 0.0,
        "ttft_full":   [float(x) for x in ttft_full],
        "ttft_single": [float(x) for x in ttft_single],
        "ttft_pair":   [float(x) for x in ttft_pair],
        "collect_seconds_full":   [float(x) for x in collect_full],
        "collect_seconds_single": [float(x) for x in collect_single],
        "collect_seconds_pair":   [float(x) for x in collect_pair],
        "total_seconds_full":   [float(x) for x in total_full],
        "total_seconds_single": [float(x) for x in total_single],
        "total_seconds_pair":   [float(x) for x in total_pair],
        "per_query_single_stats": single_query_stats,
        "per_query_pair_stats":   pair_query_stats,
        "single_fifo_kv": single_fifo_stat,
        "pair_fifo_kv": pair_fifo_stat,
        "pair_store": pair_store_stat,
        "n_queries": n,
        "dataset": str(Path(dataset_path).resolve()),
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "pair_mode": (
            "standard_per_query_no_cache" if standard_qa
            else "realistic_fifo_treat_all_cached"
        ),
        "recomp_ratio": recomp_ratio,
        "pair_recomp_ratio": pair_recomp_ratio,
        "eval": "standard_3way" if standard_qa else "realistic_3way",
    }
    with open(scores_path, "w") as f:
        json.dump(scores_payload, f, indent=2)
    print(f"3-way scores saved to {scores_path}")

    roll_raw = os.environ.get("REALISTIC_TTFT_ROLL_WINDOW", "25")
    try:
        roll_window = max(0, int(roll_raw))
    except ValueError:
        roll_window = 25
    plot_meta = {
        "single_fifo_kv": single_fifo_stat,
        "pair_fifo_kv": pair_fifo_stat,
        "pair_store": pair_store_stat,
        "pair_mode": (
            "standard_per_query_no_cache" if standard_qa
            else "realistic_fifo_treat_all_cached"
        ),
    }
    _save_3way_warmup(
        dataset_path,
        ttft_full,
        ttft_single,
        ttft_pair,
        stream_seed=stream_seed,
        skip_first=skip_first,
        metadata=plot_meta,
        roll_window=roll_window,
    )
    _save_3way_hist(
        dataset_path,
        ttft_full,
        ttft_single,
        ttft_pair,
        metadata=plot_meta,
    )

    return {
        "ttft_full": ttft_full,
        "ttft_single": ttft_single,
        "ttft_pair": ttft_pair,
        "collect_seconds_full": collect_full,
        "collect_seconds_single": collect_single,
        "collect_seconds_pair": collect_pair,
        f"{metric_name}_full":   metric_full,
        f"{metric_name}_single": metric_single,
        f"{metric_name}_pair":   metric_pair,
        "coretrieval": coret_stats,
        "per_query_single_stats": single_query_stats,
        "per_query_pair_stats":   pair_query_stats,
    }
