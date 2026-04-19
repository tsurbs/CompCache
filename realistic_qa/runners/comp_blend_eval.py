"""Shared composition-cache (CompCache) evaluation for realistic and standard QA."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SQ_RUNNERS = str(_REPO_ROOT / "standard_qa" / "runners")
if _SQ_RUNNERS not in sys.path:
    sys.path.insert(0, _SQ_RUNNERS)

from utils import CoRetrievalTracker, get_doc_ids, load_dataset  # noqa: E402
from ttft_reporting import save_ttft_histogram, save_ttft_warmup_plot  # noqa: E402

from co_retrieval_logger import CoRetrievalLogger  # noqa: E402
from composition_cache import CompositionCache  # noqa: E402
from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402
from pair_kv_store import FullJointPairStore  # noqa: E402
from promotion_worker import PromotionWorker  # noqa: E402


def _chunk_cache_key(chunk_index: int, token_ids: list[int]) -> str:
    if chunk_index == 0:
        return "__instr_prefix__"
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return f"ctx:{h.hexdigest()}"


def run_blend_eval_comp(
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
    promotion_threshold: int = 10,
    promote_sync: bool = False,
    shuffle_dataset: bool = True,
    standard_qa: bool = False,
):
    """Composition-aware CacheBlend (Proposal §3.1).

    Writes ``*_comp_coretrieval.json``, ``*_comp_scores.json``,
    ``*_comp_ttft_warmup.{json,png}``, ``*_comp_ttft_hist.{json,png}``.

    Set ``standard_qa=True`` when called from ``standard_qa/runners/utils`` so
    ``*_comp_scores.json`` includes ``"eval": "standard_comp"``.
    """
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

    individual_cache = FIFOChunkKVCache(
        fifo_max_chunks,
        store_on_cpu=standard_qa,
    )
    pair_store = FullJointPairStore(
        pair_store_capacity,
        store_on_cpu=standard_qa,
    )
    logger = CoRetrievalLogger(promotion_threshold=promotion_threshold)
    tracker = CoRetrievalTracker()

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

    worker = PromotionWorker(pair_store, _run_pair_forward, gpu_lock)
    if not promote_sync:
        worker.start()

    composition = CompositionCache(
        individual_cache=individual_cache,
        pair_store=pair_store,
        logger=logger,
        promotion_worker=worker,
        promote_sync=promote_sync,
    )

    def _chunk_key_fn(tokens):
        return _chunk_cache_key(1, tokens)

    ttft_blend = []
    ttft_full = []
    metric_blend = []
    metric_full = []
    comp_stats_per_query: list = []

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

            with gpu_lock:
                cache_fuse_metadata["collect"] = True
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v

                input_ids, chunk_past_key_values, _, qstats = composition.assemble(
                    doc_chunk_ids=doc_chunk_ids,
                    retrieval_doc_ids=doc_ids,
                    instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                    query_cache_key=f"q:{sample_idx}",
                    chunk_cache_key_fn=_chunk_key_fn,
                    run_instr_forward=_run_forward,
                    run_chunk_forward=_run_forward,
                    s_start_len=s_start_len,
                    s_start_1_len=s_start_1_len,
                )
                model_ref.old_kvs = chunk_past_key_values

                input_prompt = tokenizer.decode(input_ids)
                sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
                cache_fuse_metadata["check"] = True
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["suffix_len"] = last_len
                if recomp_ratio is not None:
                    cache_fuse_metadata["recomp_ratio"] = recomp_ratio
                if fast_attention is not None:
                    cache_fuse_metadata["fast_attention"] = fast_attention

                print(f"Sample idx: {sample_idx} stats: {qstats.as_dict()}")
                output = llm.generate([input_prompt], sampling_params)

            res = output[0].outputs[0].text
            if post_process:
                res = post_process(res)
            print(f"Cached generation: {res}")
            ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
            print(f"TTFT with cache: {ttft}")
            ttft_blend.append(ttft)
            score = max(metric_fn(res, answer, tokenizer) for answer in answers)
            metric_blend.append(score)
            comp_stats_per_query.append(qstats.as_dict())

            with gpu_lock:
                sampling_params_full = SamplingParams(temperature=0, max_tokens=max_tokens)
                cache_fuse_metadata["check"] = False
                cache_fuse_metadata["collect"] = False
                output = llm.generate([input_prompt], sampling_params_full)
            res = output[0].outputs[0].text
            if post_process:
                res = post_process(res)
            print(f"Normal generation: {res}")
            ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
            print(f"TTFT with full prefill: {ttft}")
            ttft_full.append(ttft)
            score = max(metric_fn(res, answer, tokenizer) for answer in answers)
            metric_full.append(score)
            print("------------")
    finally:
        if worker.is_alive():
            worker.stop(drain=False)
            worker.join(timeout=5.0)

    print("\n=============== Result Summary (comp) =====================")
    print(f"TTFT with cache: {np.mean(ttft_blend)}")
    print(f"TTFT with full prefill: {np.mean(ttft_full)}")
    print(f"{metric_name} with cache: {np.mean(metric_blend)}")
    print(f"{metric_name} with full prefill: {np.mean(metric_full)}")
    print(f"FIFO stats: {individual_cache.stats()}")
    print(f"Pair store: {pair_store.stats()}")
    print(f"Worker: {worker.stats()}")
    print(f"Logger: {logger.summary()}")

    fifo_stat_dict = individual_cache.stats()
    pair_stats = pair_store.stats()
    worker_stats = worker.stats()
    coret_stats = tracker.summary()
    coret_stats.update({
        "fifo_kv": fifo_stat_dict,
        "pair_store": pair_stats,
        "promotion_worker": worker_stats,
        "logger": logger.summary(),
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "promotion_threshold": promotion_threshold,
        "promote_sync": promote_sync,
    })

    output_path = dataset_path.replace(".json", "_comp_coretrieval.json")
    with open(output_path, "w") as f:
        json.dump(coret_stats, f, indent=2, default=str)
    print(f"\nComposition-cache stats saved to {output_path}")

    scores_path = Path(dataset_path).resolve()
    scores_path = scores_path.with_name(f"{scores_path.stem}_comp_scores.json")
    scores_payload = {
        "metric_name": metric_name,
        f"{metric_name}_blend": [float(x) for x in metric_blend],
        f"{metric_name}_full": [float(x) for x in metric_full],
        f"mean_{metric_name}_blend": float(np.mean(metric_blend)),
        f"mean_{metric_name}_full": float(np.mean(metric_full)),
        "ttft_blend": [float(x) for x in ttft_blend],
        "ttft_full": [float(x) for x in ttft_full],
        "per_query_stats": comp_stats_per_query,
        "fifo_kv": fifo_stat_dict,
        "pair_store": pair_stats,
        "promotion_worker": worker_stats,
        "n_queries": len(metric_blend),
        "dataset": str(Path(dataset_path).resolve()),
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "promotion_threshold": promotion_threshold,
    }
    if standard_qa:
        scores_payload["eval"] = "standard_comp"
    with open(scores_path, "w") as f:
        json.dump(scores_payload, f, indent=2)
    print(f"Quality scores saved to {scores_path}")

    roll_raw = os.environ.get("REALISTIC_TTFT_ROLL_WINDOW", "25")
    try:
        roll_window = max(0, int(roll_raw))
    except ValueError:
        roll_window = 25
    hist_meta = {
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "fifo_kv": fifo_stat_dict,
        "pair_store": pair_stats,
        "promotion_worker": worker_stats,
        "promotion_threshold": promotion_threshold,
    }
    save_ttft_warmup_plot(
        dataset_path,
        ttft_blend,
        ttft_full,
        stream_seed=stream_seed,
        skip_first=skip_first,
        fifo_stats=fifo_stat_dict,
        roll_window=roll_window,
        cached_label="CompCache (composition-aware)",
        name_suffix="_comp",
    )
    save_ttft_histogram(
        dataset_path,
        ttft_blend,
        ttft_full,
        cached_label="CompCache (composition-aware)",
        name_suffix="_comp",
        extra_metadata=hist_meta,
    )

    return {
        "ttft_blend": ttft_blend,
        "ttft_full": ttft_full,
        f"{metric_name}_blend": metric_blend,
        f"{metric_name}_full": metric_full,
        "coretrieval": coret_stats,
        "per_query_stats": comp_stats_per_query,
    }
