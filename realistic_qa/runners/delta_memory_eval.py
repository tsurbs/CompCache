from __future__ import annotations

import hashlib
import json
import math
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

@dataclass
class PairStoreConfig:
    name: str
    kind: str  
    top_k_ratio: float = 1.0  

def _parse_config(token: str) -> PairStoreConfig:
    t = token.strip().lower()
    if t in ("full", "full_joint", "joint"):
        return PairStoreConfig(name="Full (joint)", kind="full")
    if t.startswith("delta_r"):
        try:
            ratio = float(t[len("delta_r"):])
        except ValueError as exc:
            raise ValueError(f"bad config {token!r}: cannot parse ratio") from exc
        if not 0 < ratio <= 1:
            raise ValueError(f"bad config {token!r}: ratio must be in (0, 1]")
        pretty = f"Δ-sparse r={ratio:g}"
        return PairStoreConfig(name=pretty, kind="delta", top_k_ratio=ratio)
    raise ValueError(
        f"unknown config token {token!r}; expected 'full' or 'delta_r<ratio>' "
        f"(e.g. 'delta_r0.10')"
    )

def _default_configs() -> List[PairStoreConfig]:
    raw = os.environ.get("DELTA_MEMORY_CONFIGS", "full,delta_r0.50,delta_r0.10")
    return [_parse_config(t) for t in raw.split(",") if t.strip()]

@dataclass
class RunState:
    cfg: PairStoreConfig
    fifo: FIFOChunkKVCache
    pair_store: PairKVStore
    logger: CoRetrievalLogger
    worker: PromotionWorker
    composition: CompositionCache
    ttft: List[float] = field(default_factory=list)
    metric: List[float] = field(default_factory=list)
    collect_seconds: List[float] = field(default_factory=list)
    pair_hits: List[int] = field(default_factory=list)
    pair_misses: List[int] = field(default_factory=list)
    individual_hits: List[int] = field(default_factory=list)
    individual_misses: List[int] = field(default_factory=list)
    
    memory_log: List[Tuple[int, int, int]] = field(default_factory=list)

def _chunk_cache_key(chunk_index: int, token_ids: List[int]) -> str:
    if chunk_index == 0:
        return "__instr_prefix__"
    h = hashlib.sha256()
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return f"ctx:{h.hexdigest()}"

def run_delta_memory_eval(
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
    gpu_memory_utilization: float = 0.45,
    max_model_len: int | None = None,
    num_layers: int = 32,
    stream_seed: int = 34,
    skip_first: int = 0,
    fifo_max_chunks: int = 10_000,
    pair_store_capacity: int = 4096,
    pair_store_bytes_budget: Optional[int] = None,
    promotion_threshold: int = 0,
    promote_sync: bool = False,
    shuffle_dataset: bool = True,
    configs: Optional[List[PairStoreConfig]] = None,
):
    import numpy as np
    import torch
    from itertools import chain
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    configs = configs or _default_configs()
    if not configs:
        raise ValueError("at least one config required")

    full_dataset = load_dataset(dataset_path)
    if shuffle_dataset:
        rng = __import__("random").Random(stream_seed)
        full_dataset = list(full_dataset)
        rng.shuffle(full_dataset)
    eval_dataset = full_dataset[skip_first:] if skip_first else full_dataset

    llm_kwargs: dict = {
        "model": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        
        
        
        
        
        
        
        "enforce_eager": True,
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

    
    
    
    
    
    
    _empty_old_kvs = [[None, None]] * len(layers)

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

    def _make_store(cfg: PairStoreConfig) -> PairKVStore:
        
        
        
        
        
        
        if cfg.kind == "full":
            return FullJointPairStore(
                pair_store_capacity,
                bytes_budget=pair_store_bytes_budget,
                store_on_cpu=True,  
            )
        return SparseDeltaPairStore(
            pair_store_capacity,
            bytes_budget=pair_store_bytes_budget,
            top_k_ratio=cfg.top_k_ratio,
            store_on_cpu=True,
        )

    
    
    
    
    
    
    shared_fifo = FIFOChunkKVCache(fifo_max_chunks, store_on_cpu=True)

    runs: List[RunState] = []
    for cfg in configs:
        store = _make_store(cfg)
        logger = CoRetrievalLogger(promotion_threshold=promotion_threshold)
        
        
        
        
        
        
        
        
        worker = PromotionWorker(
            store, _run_pair_forward, gpu_lock, max_queue_size=64
        )
        if not promote_sync:
            worker.start()
        comp = CompositionCache(
            individual_cache=shared_fifo,
            pair_store=store,
            logger=logger,
            promotion_worker=worker,
            promote_sync=promote_sync,
        )
        runs.append(
            RunState(
                cfg=cfg,
                fifo=shared_fifo,
                pair_store=store,
                logger=logger,
                worker=worker,
                composition=comp,
            )
        )

    tracker = CoRetrievalTracker()

    base = Path(dataset_path).resolve()
    dataset_label = base.stem
    scores_path = base.with_name(f"{base.stem}_delta_memory_scores.json")

    def _write_artifacts(is_checkpoint: bool = False) -> None:
        n_local = len(runs[0].ttft) if runs else 0
        if n_local == 0:
            return

        per_config_summary_local: List[dict] = []
        for run in runs:
            peak_mb = (max((m[1] for m in run.memory_log), default=0)) / (1024.0 ** 2)
            per_config_summary_local.append({
                "name": run.cfg.name,
                "kind": run.cfg.kind,
                "top_k_ratio": run.cfg.top_k_ratio,
                "peak_mb": peak_mb,
                "final_entries": (run.memory_log[-1][2] if run.memory_log else 0),
                f"mean_{metric_name}": float(np.mean(run.metric)) if run.metric else 0.0,
                "mean_ttft_seconds": float(np.mean(run.ttft)) if run.ttft else 0.0,
                "mean_collect_seconds": (
                    float(np.mean(run.collect_seconds)) if run.collect_seconds else 0.0
                ),
                "pair_store_stats": run.pair_store.stats(),
                "fifo_stats": run.fifo.stats(),
                "worker_stats": run.worker.stats(),
                "logger_summary": run.logger.summary(),
            })

        payload_local = {
            "dataset": str(base),
            "n_queries": n_local,
            "is_checkpoint": bool(is_checkpoint),
            "stream_seed": stream_seed,
            "skip_first": skip_first,
            "pair_store_capacity": pair_store_capacity,
            "pair_store_bytes_budget": pair_store_bytes_budget,
            "promotion_threshold": promotion_threshold,
            "promote_sync": promote_sync,
            "metric_name": metric_name,
            "configs": [c.__dict__ for c in configs],
            "per_config": per_config_summary_local,
            "per_query": {
                run.cfg.name: {
                    "ttft": [float(x) for x in run.ttft],
                    "collect_seconds": [float(x) for x in run.collect_seconds],
                    metric_name: [float(x) for x in run.metric],
                    "pair_hits": run.pair_hits,
                    "pair_misses": run.pair_misses,
                    "individual_hits": run.individual_hits,
                    "individual_misses": run.individual_misses,
                    "memory_log": [[i, int(b), int(e)] for (i, b, e) in run.memory_log],
                }
                for run in runs
            },
            "coretrieval": tracker.summary(),
        }

        
        
        tmp_scores = scores_path.with_suffix(scores_path.suffix + ".tmp")
        with open(tmp_scores, "w") as f:
            json.dump(payload_local, f, indent=2, default=str)
        os.replace(tmp_scores, scores_path)

        tag = "checkpoint" if is_checkpoint else "final"
        print(
            f"[delta-mem] {tag} artifacts written @ n={n_local}: "
            f"{scores_path.name}",
            flush=True,
        )

    def _run_cached_prefill(input_prompt: str, last_len: int) -> Tuple[float, str]:
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["suffix_len"] = last_len
        if recomp_ratio is not None:
            cache_fuse_metadata["recomp_ratio"] = recomp_ratio
        if fast_attention is not None:
            cache_fuse_metadata["fast_attention"] = fast_attention
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

            for run in runs:
                with gpu_lock:
                    cache_fuse_metadata["collect"] = True
                    cache_fuse_metadata["check"] = False
                    for k, v in extra_metadata.items():
                        cache_fuse_metadata[k] = v

                    t0 = time.perf_counter()
                    input_ids, chunk_past_key_values, _, qstats = run.composition.assemble(
                        doc_chunk_ids=doc_chunk_ids,
                        retrieval_doc_ids=doc_ids,
                        instr_cache_key=_chunk_cache_key(0, doc_chunk_ids[0]),
                        query_cache_key=f"q:{run.cfg.name}:{sample_idx}",
                        chunk_cache_key_fn=lambda t: _chunk_cache_key(1, t),
                        run_instr_forward=_run_forward,
                        run_chunk_forward=_run_forward,
                        s_start_len=s_start_len,
                        s_start_1_len=s_start_1_len,
                    )
                    collect_s = time.perf_counter() - t0
                    model_ref.old_kvs = chunk_past_key_values
                    input_prompt = tokenizer.decode(input_ids)
                    ttft, text = _run_cached_prefill(input_prompt, last_len)

                run.ttft.append(ttft)
                run.collect_seconds.append(collect_s)
                run.pair_hits.append(qstats.pair_hits)
                run.pair_misses.append(qstats.pair_misses)
                run.individual_hits.append(qstats.individual_hits)
                run.individual_misses.append(qstats.individual_misses)
                score = max(metric_fn(text, ans, tokenizer) for ans in answers)
                run.metric.append(score)
                run.memory_log.append((
                    sample_idx,
                    run.pair_store.bytes_used(),
                    len(run.pair_store._store),  # type: ignore[attr-defined]
                ))

                
                
                
                
                
                
                
                
                model_ref.old_kvs = _empty_old_kvs
                del chunk_past_key_values
                del input_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if sample_idx % 10 == 0 or sample_idx == len(eval_dataset) - 1:
                parts = []
                for run in runs:
                    mb = run.memory_log[-1][1] / (1024.0 ** 2)
                    parts.append(f"{run.cfg.name}: {mb:.1f}MB, {run.pair_store.stats()['entries']}e")
                print(f"[delta-mem idx={sample_idx}] " + " | ".join(parts), flush=True)

            
            
            
            completed = sample_idx + 1
            if completed == 50 or (completed >= 100 and completed % 100 == 0):
                _write_artifacts(is_checkpoint=True)
    finally:
        for run in runs:
            if run.worker.is_alive():
                run.worker.stop(drain=False)
                run.worker.join(timeout=5.0)
        
        
        
        try:
            _write_artifacts(is_checkpoint=False)
        except Exception as e:  
            print(f"[delta-mem] final artifact flush failed: {e!r}", flush=True)

    
    
    
    n = len(runs[0].ttft) if runs else 0
    try:
        with open(scores_path, "r") as f:
            final_payload = json.load(f)
        per_config_summary = final_payload.get("per_config", [])
    except Exception:
        per_config_summary = []

    return {
        "n_queries": n,
        "per_config": per_config_summary,
        "artifacts": {
            "scores_json": str(scores_path),
            "timeseries_png": None,
            "tradeoff_png": None,
            "ttft_png": None,
        },
    }
