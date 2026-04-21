"""Recomputation-ratio sweep: evaluate CompCache(single) and CompCache(+pairs) at
a range of ``recomp_ratio`` values on the same query stream, holding everything
else constant.

For each query we:

1. Run Full Prefill **once** (independent of ``recomp_ratio``).
2. Build the single-chunk cached KV stack **once** (the "collect" phase).
3. Build the pair-aware cached KV stack **once**.
4. Loop over the supplied ``recomp_ratios`` and, for each value ``r``, run the
   cached prefill twice — once with the single KV stack at ``recomp_ratio=r``
   and once with the pair KV stack at ``recomp_ratio=r/2`` (pair repair budget
   is always half the single one, per user convention).

This keeps expensive chunk-level forwards + KV copies out of the inner sweep
loop — we only pay the (fast) cached-prefill pass repeatedly, which is what
actually depends on ``recomp_ratio``. Compared to rerunning the full 3-way
evaluation eight times (8× model load, 8× collect) this runs ~8× faster.

Outputs one JSON file ``{stem}_recomp_sweep_scores.json`` (set env var
``RECOMP_SWEEP_OUTPUT_TAG`` to override the suffix) with arrays of per-query
metric/TTFT values for every method and every ratio.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SQ_RUNNERS = str(_REPO_ROOT / "standard_qa" / "runners")
if _SQ_RUNNERS not in sys.path:
    sys.path.insert(0, _SQ_RUNNERS)

from utils import CoRetrievalTracker, get_doc_ids, load_dataset  # noqa: E402

from co_retrieval_logger import CoRetrievalLogger  # noqa: E402
from composition_cache import CompositionCache  # noqa: E402
from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402
from pair_kv_store import FullJointPairStore  # noqa: E402
from per_query_pair_assembler import assemble_pairs_per_query  # noqa: E402


def _chunk_cache_key(chunk_index: int, token_ids: List[int]) -> str:
    if chunk_index == 0:
        return "__instr_prefix__"
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return f"ctx:{h.hexdigest()}"


def _sweep_suffix() -> str:
    tag = os.environ.get("RECOMP_SWEEP_OUTPUT_TAG", "").strip()
    return f"_recomp_sweep_{tag}" if tag else "_recomp_sweep"


def run_recomp_sweep_eval(
    dataset_path: str,
    prefix_prompt: str,
    prompt_builder,
    metric_fn,
    metric_name: str,
    *,
    recomp_ratios: Sequence[float],
    pair_ratio_fn: Callable[[float], float] = lambda r: r / 2.0,
    inst_tokens=None,
    s_end=None,
    suffix_is_query_len: bool = True,
    max_ctx_len=None,
    max_tokens: int = 32,
    fast_attention=None,
    extra_metadata=None,
    post_process=None,
    clear_hack_kv: bool = False,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization: float = 0.38,
    max_model_len: int | None = None,
    num_layers: int = 32,
    fifo_max_chunks: int = 512,
    shuffle_dataset: bool = False,
):
    """Sweep eval: Full once, single KVs once, pair KVs once, then cached-prefill
    at each ``recomp_ratio`` for both single and pair KVs.

    No cross-query caching anywhere (mirrors ``standard_qa`` 3-way mode): every
    query re-builds its own single FIFO content via ``CompositionCache.assemble``
    and its own per-query pair KVs via ``assemble_pairs_per_query``.
    """
    import numpy as np
    import torch  # noqa: F401
    from itertools import chain
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    recomp_ratios = [float(r) for r in recomp_ratios]
    pair_ratios = [float(pair_ratio_fn(r)) for r in recomp_ratios]
    assert recomp_ratios, "recomp_ratios must be non-empty"

    full_dataset = load_dataset(dataset_path)
    if shuffle_dataset:
        rng = __import__("random").Random(0)
        full_dataset = list(full_dataset)
        rng.shuffle(full_dataset)

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
        f"[sweep] recomp_ratios(single)={recomp_ratios}\n"
        f"[sweep] recomp_ratios(+pairs)={pair_ratios}",
        flush=True,
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

    def _chunk_key_fn(tokens):
        return _chunk_cache_key(1, tokens)

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

    def _run_cached_prefill(input_prompt: str, last_len: int, recomp: float):
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["suffix_len"] = last_len
        cache_fuse_metadata["recomp_ratio"] = recomp
        if fast_attention is not None:
            cache_fuse_metadata["fast_attention"] = fast_attention
        out = llm.generate([input_prompt], sp)
        ttft = out[0].metrics.first_token_time - out[0].metrics.first_scheduled_time
        text = out[0].outputs[0].text
        if post_process:
            text = post_process(text)
        return ttft, text

    # ---- Accumulators ------------------------------------------------------
    # full: one score/ttft per query
    ttft_full: list[float] = []
    metric_full: list[float] = []
    collect_full: list[float] = []

    # single/pair: per (ratio, query) score/ttft
    n_r = len(recomp_ratios)
    ttft_single: list[list[float]] = [[] for _ in range(n_r)]
    ttft_pair:   list[list[float]] = [[] for _ in range(n_r)]
    metric_single: list[list[float]] = [[] for _ in range(n_r)]
    metric_pair:   list[list[float]] = [[] for _ in range(n_r)]
    # collect times are independent of ratio (shared across the inner loop)
    collect_single: list[float] = []
    collect_pair: list[float] = []

    try:
        for sample_idx, ex in enumerate(full_dataset):
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

            # ---- Full Prefill (once) ----
            with gpu_lock:
                cache_fuse_metadata["collect"] = False
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t_full, text_full = _run_full_prefill(full_input_prompt)
            score_full = max(metric_fn(text_full, ans, tokenizer) for ans in answers)
            ttft_full.append(t_full)
            collect_full.append(0.0)
            metric_full.append(score_full)
            print(
                f"[sweep idx={sample_idx}] full TTFT={t_full:.4f}s score={score_full:.3f} -> {text_full!r}",
                flush=True,
            )

            # ---- Build single KVs (once, no cross-query cache) ----
            # Use a fresh per-query FIFO so no pair state leaks in. We still use
            # CompositionCache(disable_pairs=True) so the tokenization / prompt
            # assembly exactly matches the 3-way single path.
            per_query_fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=cache_store_on_cpu)
            composition_single = CompositionCache(
                individual_cache=per_query_fifo,
                pair_store=null_pair_store,
                logger=single_logger,
                promotion_worker=None,
                promote_sync=False,
            )
            with gpu_lock:
                cache_fuse_metadata["collect"] = True
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t0 = time.perf_counter()
                input_ids_s, kvs_s, _, _qstats_s = composition_single.assemble(
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
                coll_s = time.perf_counter() - t0
            collect_single.append(coll_s)
            input_prompt_s = tokenizer.decode(input_ids_s)

            # ---- Build pair KVs (once, no cross-query cache) ----
            with gpu_lock:
                cache_fuse_metadata["collect"] = True
                cache_fuse_metadata["check"] = False
                for k, v in extra_metadata.items():
                    cache_fuse_metadata[k] = v
                t0 = time.perf_counter()
                input_ids_p, kvs_p, _, _qstats_p = assemble_pairs_per_query(
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
                coll_p = time.perf_counter() - t0
            collect_pair.append(coll_p)
            input_prompt_p = tokenizer.decode(input_ids_p)

            # ---- Sweep cached-prefill over recomp ratios ----
            for ri, (r_single, r_pair) in enumerate(zip(recomp_ratios, pair_ratios)):
                with gpu_lock:
                    for k, v in extra_metadata.items():
                        cache_fuse_metadata[k] = v
                    model_ref.old_kvs = kvs_s
                    t_s, text_s = _run_cached_prefill(input_prompt_s, last_len, r_single)
                    model_ref.old_kvs = kvs_p
                    t_p, text_p = _run_cached_prefill(input_prompt_p, last_len, r_pair)
                sc_s = max(metric_fn(text_s, ans, tokenizer) for ans in answers)
                sc_p = max(metric_fn(text_p, ans, tokenizer) for ans in answers)
                ttft_single[ri].append(t_s)
                ttft_pair[ri].append(t_p)
                metric_single[ri].append(sc_s)
                metric_pair[ri].append(sc_p)
                print(
                    f"[sweep idx={sample_idx}] r_s={r_single:.3f} r_p={r_pair:.3f} "
                    f"single TTFT={t_s:.4f}s score={sc_s:.3f} | "
                    f"+pairs TTFT={t_p:.4f}s score={sc_p:.3f}",
                    flush=True,
                )
            print("------------", flush=True)
    finally:
        pass

    # ---- Save ---------------------------------------------------------------
    suffix = _sweep_suffix()
    n_q = len(ttft_full)
    base = Path(dataset_path).resolve()
    out_path = base.with_name(f"{base.stem}{suffix}_scores.json")

    # Per-ratio means
    per_ratio = []
    import statistics as _s
    for ri, r in enumerate(recomp_ratios):
        per_ratio.append({
            "recomp_ratio_single": r,
            "recomp_ratio_pair":   pair_ratios[ri],
            "mean_ttft_single":    float(_s.mean(ttft_single[ri]))    if ttft_single[ri]    else 0.0,
            "mean_ttft_pair":      float(_s.mean(ttft_pair[ri]))      if ttft_pair[ri]      else 0.0,
            f"mean_{metric_name}_single": float(_s.mean(metric_single[ri])) if metric_single[ri] else 0.0,
            f"mean_{metric_name}_pair":   float(_s.mean(metric_pair[ri]))   if metric_pair[ri]   else 0.0,
            "ttft_single": [float(x) for x in ttft_single[ri]],
            "ttft_pair":   [float(x) for x in ttft_pair[ri]],
            f"{metric_name}_single": [float(x) for x in metric_single[ri]],
            f"{metric_name}_pair":   [float(x) for x in metric_pair[ri]],
        })

    payload = {
        "eval": "recomp_sweep",
        "dataset": str(base),
        "metric_name": metric_name,
        "n_queries": n_q,
        "recomp_ratios_single": recomp_ratios,
        "recomp_ratios_pair":   pair_ratios,
        "pair_is_half_of_single": True,
        "mean_ttft_full":  float(_s.mean(ttft_full))  if ttft_full  else 0.0,
        f"mean_{metric_name}_full": float(_s.mean(metric_full)) if metric_full else 0.0,
        "mean_collect_single": float(_s.mean(collect_single)) if collect_single else 0.0,
        "mean_collect_pair":   float(_s.mean(collect_pair))   if collect_pair   else 0.0,
        "ttft_full":   [float(x) for x in ttft_full],
        f"{metric_name}_full":   [float(x) for x in metric_full],
        "collect_single": [float(x) for x in collect_single],
        "collect_pair":   [float(x) for x in collect_pair],
        "per_ratio": per_ratio,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSweep scores saved to {out_path}", flush=True)

    # ---- Console summary ----------------------------------------------------
    print("\n========= Recomp sweep summary =========")
    print(f"n_queries = {n_q}")
    print(f"Full  TTFT={payload['mean_ttft_full']:.4f}s  {metric_name}={payload[f'mean_{metric_name}_full']:.3f}")
    print(f"{'r_single':>9} {'r_pair':>8} {'TTFT_s':>9} {'TTFT_p':>9} "
          f"{metric_name+'_s':>9} {metric_name+'_p':>9}")
    for r in per_ratio:
        print(f"{r['recomp_ratio_single']:>9.3f} {r['recomp_ratio_pair']:>8.3f} "
              f"{r['mean_ttft_single']:>9.4f} {r['mean_ttft_pair']:>9.4f} "
              f"{r[f'mean_{metric_name}_single']:>9.3f} {r[f'mean_{metric_name}_pair']:>9.3f}")
    return payload
