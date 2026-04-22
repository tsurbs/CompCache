"""Test 2: revamped "realistic" eval with an LFU-by-pair pair store, where
the memory savings of :class:`SparseDeltaPairStore` are spent on caching
MORE pairs at the same bytes budget as :class:`FullJointPairStore`.

Each query streams through four methods on a shared vLLM engine (each
with its own FIFO / pair store / logger so state is honest):

1. ``full``   — vanilla prefill, no caching (upper bound on TTFT, ground truth for F1).
2. ``single`` — CompCache with the pair path disabled (CacheBlend single-chunk).
3. ``lfu_full``  — CompCache + :class:`FullJointPairStore` with
   ``evict_policy="lfu"``; priority is the running co-retrieval count
   from a per-method :class:`CoRetrievalLogger`.  Capacity: ``cap_full``.
4. ``lfu_delta`` — CompCache + :class:`SparseDeltaPairStore` with
   ``top_k_ratio=r`` and the same LFU priority policy.  Capacity:
   ``cap_delta ≈ cap_full / r`` so the asymptotic BYTES footprint
   matches method 3 — we spend the delta savings on more entries, not
   on less memory.

Artifacts written next to the dataset:

- ``{stem}_budget_scores.json`` — per-method per-query series + summary.
- ``{stem}_budget_main.png`` — 2x2 grid: rolling TTFT, cumulative pair
  hit rate, mean metric (bar), bytes_used over time.
- ``{stem}_budget_ttft_hist.png`` — TTFT density overlay, all four methods.
"""
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

from utils import CoRetrievalTracker, get_doc_ids, load_dataset  # noqa: E402

from co_retrieval_logger import CoRetrievalLogger  # noqa: E402
from composition_cache import CompositionCache  # noqa: E402
from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402
from pair_kv_store import (  # noqa: E402
    FullJointPairStore,
    PairKVStore,
    SparseDeltaPairStore,
)
from promotion_worker import PromotionWorker  # noqa: E402


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
    # Filled only for LFU methods
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


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def _rolling_mean(xs, w: int):
    import numpy as np
    a = np.asarray(xs, dtype=float)
    if w <= 1 or a.size < w:
        return np.arange(a.size, dtype=float), a
    k = np.ones(w, dtype=float) / w
    y = np.convolve(a, k, mode="valid")
    x = np.arange(w - 1, a.size, dtype=float)
    return x, y


def _plot_main_grid(
    output_png: Path,
    states: dict[str, MethodState],
    *,
    metric_name: str,
    roll_window: int,
    dataset_label: str,
    budget_mb_target: Optional[float],
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), layout="constrained")
    ax_tt, ax_hr, ax_f1, ax_mb = axes.flatten()

    # ---- Panel 1: rolling TTFT ----
    for m in METHODS:
        st = states[m]
        if not st.ttft:
            continue
        color = METHOD_COLORS[m]
        raw_ms = np.asarray(st.ttft) * 1e3
        ax_tt.plot(
            np.arange(raw_ms.size), raw_ms,
            color=color, lw=0.6, alpha=0.28,
        )
        xs, ys = _rolling_mean(raw_ms, roll_window)
        ax_tt.plot(
            xs, ys,
            color=color, lw=2.2,
            label=f"{METHOD_LABELS[m]}  (mean {raw_ms.mean():.1f} ms)",
        )
    ax_tt.set_xlabel("Query index")
    ax_tt.set_ylabel("Prefill TTFT (ms)")
    ax_tt.set_title(f"TTFT — rolling {roll_window}-query mean")
    ax_tt.grid(True, alpha=0.3)
    ax_tt.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- Panel 2: cumulative pair hit rate ----
    for m in (METHOD_LFU_FULL, METHOD_LFU_DELTA):
        st = states[m]
        if not st.pair_hits:
            continue
        hits = np.asarray(st.pair_hits, dtype=float)
        misses = np.asarray(st.pair_misses, dtype=float)
        # Cumulative hits / (hits + misses). Division-by-zero guarded.
        num = np.cumsum(hits)
        den = np.cumsum(hits + misses)
        rate = np.where(den > 0, num / np.maximum(den, 1.0), 0.0) * 100.0
        ax_hr.plot(
            np.arange(rate.size), rate,
            color=METHOD_COLORS[m], lw=2.2,
            label=f"{METHOD_LABELS[m]}  (final {rate[-1]:.1f}%)" if rate.size else METHOD_LABELS[m],
        )
    ax_hr.set_xlabel("Query index")
    ax_hr.set_ylabel("Cumulative pair hit rate (%)")
    ax_hr.set_title("Pair-store effectiveness over stream")
    ax_hr.grid(True, alpha=0.3)
    ax_hr.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ---- Panel 3: mean metric bar chart ----
    names, vals, errs, colors = [], [], [], []
    for m in METHODS:
        st = states[m]
        if not st.metric:
            continue
        arr = np.asarray(st.metric, dtype=float)
        names.append(METHOD_LABELS[m].replace("CompCache + pairs ", "").replace("CompCache ", ""))
        vals.append(arr.mean())
        errs.append(arr.std(ddof=1) / max(1.0, np.sqrt(arr.size)) * 1.96)
        colors.append(METHOD_COLORS[m])
    x = np.arange(len(names))
    bars = ax_f1.bar(x, vals, yerr=errs, color=colors, edgecolor="black", capsize=4)
    for bar, v in zip(bars, vals):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2, v + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax_f1.set_ylabel(f"Mean {metric_name}  (± 95% CI)")
    ax_f1.set_title(f"Answer quality — {metric_name}")
    ax_f1.grid(True, alpha=0.3, axis="y")

    # ---- Panel 4: bytes used over time ----
    for m in (METHOD_LFU_FULL, METHOD_LFU_DELTA):
        st = states[m]
        if not st.memory_log:
            continue
        idx = np.array([e[0] for e in st.memory_log])
        mb = np.array([e[1] for e in st.memory_log], dtype=float) / (1024.0 ** 2)
        entries = np.array([e[2] for e in st.memory_log], dtype=int)
        ax_mb.plot(
            idx, mb,
            color=METHOD_COLORS[m], lw=2.0,
            label=f"{METHOD_LABELS[m]}  (peak {mb.max():.1f} MB, final n={int(entries[-1])})",
        )
    if budget_mb_target is not None and budget_mb_target > 0:
        ax_mb.axhline(
            budget_mb_target,
            color="black", ls="--", lw=1.5, alpha=0.6,
            label=f"Target bytes budget ({budget_mb_target:.1f} MB)",
        )
    ax_mb.set_xlabel("Query index")
    ax_mb.set_ylabel("Pair store size (MB)")
    ax_mb.set_title("Pair store memory footprint (budget parity)")
    ax_mb.grid(True, alpha=0.3)
    ax_mb.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Popularity-budget realistic eval — {dataset_label}",
        fontsize=13, y=1.01,
    )
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def _plot_ttft_hist(
    output_png: Path,
    states: dict[str, MethodState],
    *,
    dataset_label: str,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    all_ms = []
    for m in METHODS:
        st = states[m]
        if st.ttft:
            all_ms.append(np.asarray(st.ttft) * 1e3)
    if not all_ms:
        return None
    edges = np.histogram_bin_edges(np.concatenate(all_ms), bins="auto")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), layout="constrained")
    for m in METHODS:
        st = states[m]
        if not st.ttft:
            continue
        ms = np.asarray(st.ttft) * 1e3
        ax.hist(
            ms, bins=edges, alpha=0.45, density=True,
            color=METHOD_COLORS[m],
            label=f"{METHOD_LABELS[m]}  (mean {ms.mean():.1f} ms)",
        )
    ax.set_xlabel("Prefill TTFT (ms)")
    ax.set_ylabel("Density")
    ax.set_title(f"TTFT distribution — {dataset_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


# --------------------------------------------------------------------------- #
# Main runner                                                                 #
# --------------------------------------------------------------------------- #


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
    """Run the four methods back-to-back per query.

    ``cap_delta`` is auto-set to ``round(cap_full / delta_top_k_ratio)``
    when ``None`` so the two pair methods converge to the same asymptotic
    bytes budget.  Pass an explicit value to override.
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

    # ---- per-method state ----
    states: dict[str, MethodState] = {m: MethodState(name=m) for m in METHODS}

    # Single-chunk CompCache (no pair path).
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

    # LFU pair methods.  The priority function consults THIS method's
    # CoRetrievalLogger, not a shared one, so evictions reflect the
    # frequencies that THIS store has actually observed.
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

            # ---- Method 1: Full Prefill ----
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

            # ---- Methods 2-4: compcache variants ----
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

    # ---- artifacts --------------------------------------------------------
    n = len(states[METHOD_FULL].ttft)
    dataset_label = Path(dataset_path).stem

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

    # Budget target line: mean bytes/entry of the Full-LFU run × its
    # steady-state entry count.  This gives a human-readable "budget"
    # reference on the memory panel.
    full_state = states[METHOD_LFU_FULL]
    if full_state.memory_log:
        budget_mb = max(e[1] for e in full_state.memory_log) / (1024.0 ** 2)
    else:
        budget_mb = None

    base = Path(dataset_path).resolve()
    scores_path = base.with_name(f"{base.stem}_budget_scores.json")
    main_png = base.with_name(f"{base.stem}_budget_main.png")
    hist_png = base.with_name(f"{base.stem}_budget_ttft_hist.png")

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

    roll_raw = os.environ.get("REALISTIC_TTFT_ROLL_WINDOW", "25")
    try:
        roll_window = max(1, int(roll_raw))
    except ValueError:
        roll_window = 25

    _plot_main_grid(
        main_png, states,
        metric_name=metric_name,
        roll_window=roll_window,
        dataset_label=dataset_label,
        budget_mb_target=budget_mb,
    )
    _plot_ttft_hist(hist_png, states, dataset_label=dataset_label)
    print(f"[budget] main 2x2 plot:  {main_png}")
    print(f"[budget] TTFT hist plot: {hist_png}")

    return {
        "n_queries": n,
        "per_method_summary": summary,
        "artifacts": {
            "scores_json": str(scores_path),
            "main_png": str(main_png),
            "ttft_hist_png": str(hist_png),
        },
    }
