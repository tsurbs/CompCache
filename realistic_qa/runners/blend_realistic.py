"""CacheBlend eval with FIFO reuse of independently collected chunk KVs.

Expects a dataset JSON built by ``realistic_qa/scripts/build_extended_dataset.py``.
Processing order is shuffled (default seed 34) to mimic streaming queries.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "standard_qa" / "runners"))
sys.path.insert(0, str(_REPO / "realistic_qa" / "runners"))

from ttft_reporting import save_ttft_histogram, save_ttft_warmup_plot  # noqa: E402
from utils import (  # noqa: E402
    CoRetrievalTracker,
    build_qa_prompt,
    compute_f1,
    get_doc_ids,
    load_dataset,
)

from kv_fifo_cache import FIFOChunkKVCache  # noqa: E402

from comp_blend_eval import _chunk_cache_key, run_blend_eval_comp  # noqa: E402

query_prompt = (
    "\n\nAnswer the question directly based on the given passages."
    " Do NOT repeat the question."
    " The answer should be within 5 words. \nQuestion:"
)


def run_blend_eval_fifo(
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
    shuffle_dataset: bool = True,
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

    warmup_dataset = full_dataset[:skip_first] if skip_first else []
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

    fifo = FIFOChunkKVCache(fifo_max_chunks)
    tracker = CoRetrievalTracker()
    ttft_blend = []
    ttft_full = []
    metric_blend = []
    metric_full = []

    model_ref = (
        llm.llm_engine.model_executor.driver_worker
        .model_runner.model.model
    )
    layers = model_ref.layers

    if warmup_dataset:
        print(f"[warmup] Processing {len(warmup_dataset)} queries to populate FIFO cache …")
        sampling_params_warmup = SamplingParams(temperature=0, max_tokens=1)
        cache_fuse_metadata = model_ref.cache_fuse_metadata
        for wi, ex in enumerate(warmup_dataset):
            doc_prompts_w, _ = prompt_builder(ex)
            doc_chunk_ids_w = [tokenizer.encode(doc)[1:] for doc in doc_prompts_w]

            if max_ctx_len is not None:
                while len(list(chain.from_iterable(doc_chunk_ids_w))) > max_ctx_len:
                    del_idx = int(len(doc_chunk_ids_w) / 2)
                    del doc_chunk_ids_w[del_idx]
                if len(doc_chunk_ids_w) == 0:
                    continue

            doc_chunk_ids_w = [s_start + cids for cids in doc_chunk_ids_w]
            doc_chunk_ids_w = [s_start_full] + doc_chunk_ids_w

            cache_fuse_metadata["collect"] = True
            cache_fuse_metadata["check"] = False
            for k, v in extra_metadata.items():
                cache_fuse_metadata[k] = v

            for i, chunk_ids in enumerate(doc_chunk_ids_w):
                ck = _chunk_cache_key(i, chunk_ids)
                if fifo.get(ck) is not None:
                    continue
                prompts = [tokenizer.decode(chunk_ids)]
                llm.generate(prompts, sampling_params_warmup)
                layer_kvs: list = []
                for j in range(num_layers):
                    past_key_values = layers[j].self_attn.hack_kv
                    if i == 0:
                        temp_k = past_key_values[0][:s_start_len].clone()
                        temp_v = past_key_values[1][:s_start_len].clone()
                    else:
                        temp_k = past_key_values[0][s_start_1_len:len(chunk_ids)+1].clone()
                        temp_v = past_key_values[1][s_start_1_len:len(chunk_ids)+1].clone()
                    layer_kvs.append([temp_k, temp_v])
                    if clear_hack_kv:
                        layers[j].self_attn.hack_kv = None
                fifo.put(ck, layer_kvs)

            if (wi + 1) % 200 == 0 or wi == len(warmup_dataset) - 1:
                print(f"[warmup] {wi + 1}/{len(warmup_dataset)} — FIFO: {fifo.stats()}")

        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["check"] = False
        print(f"[warmup] Done. FIFO: {fifo.stats()}")

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
            if len(doc_chunk_ids) == 0:
                continue

        sampling_params = SamplingParams(temperature=0, max_tokens=1)

        cache_fuse_metadata = model_ref.cache_fuse_metadata
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["check"] = False
        for k, v in extra_metadata.items():
            cache_fuse_metadata[k] = v

        doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
        doc_chunk_ids = [s_start_full] + doc_chunk_ids
        doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

        last_len = len(q_ids + s_end) if suffix_is_query_len else len([q_ids + s_end])

        chunk_past_key_values: list = []

        cache_fuse_metadata["collect"] = True
        cache_fuse_metadata["check"] = False

        for i in range(len(doc_chunk_ids)):
            ck = _chunk_cache_key(i, doc_chunk_ids[i])
            cached_layers = fifo.get(ck)

            if cached_layers is not None:
                for j in range(num_layers):
                    temp_k, temp_v = cached_layers[j]
                    if i == 0:
                        chunk_past_key_values.append([temp_k.clone(), temp_v.clone()])
                    else:
                        chunk_past_key_values[j][0] = torch.cat(
                            (chunk_past_key_values[j][0], temp_k), dim=0
                        )
                        chunk_past_key_values[j][1] = torch.cat(
                            (chunk_past_key_values[j][1], temp_v), dim=0
                        )
                model_ref.old_kvs = chunk_past_key_values
                continue

            prompts = [tokenizer.decode(doc_chunk_ids[i])]
            llm.generate(prompts, sampling_params)

            layer_kvs: list = []
            for j in range(num_layers):
                past_key_values = layers[j].self_attn.hack_kv
                if i == 0:
                    temp_k = past_key_values[0][:s_start_len].clone()
                    temp_v = past_key_values[1][:s_start_len].clone()
                else:
                    temp_k = past_key_values[0][
                        s_start_1_len : len(doc_chunk_ids[i]) + 1
                    ].clone()
                    temp_v = past_key_values[1][
                        s_start_1_len : len(doc_chunk_ids[i]) + 1
                    ].clone()

                if i == 0:
                    chunk_past_key_values.append([temp_k, temp_v])
                else:
                    chunk_past_key_values[j][0] = torch.cat(
                        (chunk_past_key_values[j][0], temp_k), dim=0
                    )
                    chunk_past_key_values[j][1] = torch.cat(
                        (chunk_past_key_values[j][1], temp_v), dim=0
                    )
                layer_kvs.append([temp_k, temp_v])

                if clear_hack_kv:
                    layers[j].self_attn.hack_kv = None

            fifo.put(ck, layer_kvs)
            model_ref.old_kvs = chunk_past_key_values

        input_ids: list = []
        for i in range(len(doc_chunk_ids)):
            if i == 0:
                temp_ids = doc_chunk_ids[i]
            else:
                temp_ids = doc_chunk_ids[i][s_start_1_len - 1 :]
            input_ids += temp_ids
        input_prompt = tokenizer.decode(input_ids)

        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata["collect"] = False
        cache_fuse_metadata["suffix_len"] = last_len
        if recomp_ratio is not None:
            cache_fuse_metadata["recomp_ratio"] = recomp_ratio
        if fast_attention is not None:
            cache_fuse_metadata["fast_attention"] = fast_attention

        print(f"Sample idx: {sample_idx}")
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

        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata["check"] = False
        cache_fuse_metadata["collect"] = False
        output = llm.generate([input_prompt], sampling_params)
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

    print("\n=============== Result Summary =====================")
    print(f"TTFT with cache: {np.mean(ttft_blend)}")
    print(f"TTFT with full prefill: {np.mean(ttft_full)}")
    print(f"{metric_name} with cache: {np.mean(metric_blend)}")
    print(f"{metric_name} with full prefill: {np.mean(metric_full)}")
    print(f"FIFO stats: {fifo.stats()}")

    coret_stats = tracker.summary()
    fifo_stat_dict = fifo.stats()
    coret_stats["fifo_kv"] = fifo_stat_dict
    coret_stats["stream_seed"] = stream_seed
    coret_stats["skip_first"] = skip_first

    roll_raw = os.environ.get("REALISTIC_TTFT_ROLL_WINDOW", "25")
    try:
        roll_window = max(0, int(roll_raw))
    except ValueError:
        roll_window = 25
    save_ttft_warmup_plot(
        dataset_path,
        ttft_blend,
        ttft_full,
        stream_seed=stream_seed,
        skip_first=skip_first,
        fifo_stats=fifo_stat_dict,
        roll_window=roll_window,
    )
    save_ttft_histogram(
        dataset_path,
        ttft_blend,
        ttft_full,
        cached_label="CacheBlend (FIFO KV)",
        extra_metadata={
            "stream_seed": stream_seed,
            "skip_first": skip_first,
            "fifo_kv": fifo_stat_dict,
        },
    )

    output_path = dataset_path.replace(".json", "_coretrieval.json")
    with open(output_path, "w") as f:
        json.dump(coret_stats, f, indent=2, default=str)
    print(f"\nCo-retrieval + FIFO stats saved to {output_path}")

    scores_path = Path(dataset_path).resolve()
    scores_path = scores_path.with_name(f"{scores_path.stem}_scores.json")
    scores_payload = {
        "metric_name": metric_name,
        f"{metric_name}_blend": [float(x) for x in metric_blend],
        f"{metric_name}_full": [float(x) for x in metric_full],
        f"mean_{metric_name}_blend": float(np.mean(metric_blend)),
        f"mean_{metric_name}_full": float(np.mean(metric_full)),
        "n_queries": len(metric_blend),
        "dataset": str(Path(dataset_path).resolve()),
        "stream_seed": stream_seed,
        "skip_first": skip_first,
        "fifo_kv": fifo_stat_dict,
    }
    with open(scores_path, "w") as f:
        json.dump(scores_payload, f, indent=2)
    print(f"Quality scores saved to {scores_path}")

    return {
        "ttft_blend": ttft_blend,
        "ttft_full": ttft_full,
        f"{metric_name}_blend": metric_blend,
        f"{metric_name}_full": metric_full,
        "coretrieval": coret_stats,
    }


def _resolve_dataset_path(raw: str) -> str:
    """Paths are relative to the CompCache repo root (works on Modal cwd=/CompCache or elsewhere)."""
    p = Path(raw)
    if not p.is_absolute():
        p = (_REPO / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {p}"
        )
    return str(p)


def main() -> None:
    fifo_max = int(os.environ.get("REALISTIC_FIFO_MAX", "10000"))
    skip = int(os.environ.get("REALISTIC_SKIP_FIRST", "0"))
    dataset = os.environ.get(
        "REALISTIC_DATASET",
        "realistic_qa/inputs/extended_tiny.json",
    )
    dataset_path = _resolve_dataset_path(dataset)

    gpu_raw = os.environ.get("REALISTIC_GPU_MEMORY_UTILIZATION")
    gpu_memory_utilization = (
        float(gpu_raw) if gpu_raw is not None else 0.45
    )
    max_ctx_raw = os.environ.get("REALISTIC_MAX_CTX_LEN")
    max_ctx_len = int(max_ctx_raw) if max_ctx_raw else None
    mml_raw = os.environ.get("REALISTIC_MAX_MODEL_LEN")
    max_model_len = int(mml_raw) if mml_raw else None
    if max_model_len is None and max_ctx_len is not None:
        # Doc tokens are capped at max_ctx_len; prompt adds inst/query/template overhead.
        max_model_len = max_ctx_len + 2048

    mode = os.environ.get("REALISTIC_MODE", "fifo").lower()
    common_kwargs = dict(
        dataset_path=dataset_path,
        prefix_prompt=(
            "You will be asked a question after reading several passages."
            " Please directly answer the question based on the given passages."
            " Do NOT repeat the question."
            " The answer should be within 5 words..\nPassages:\n"
        ),
        prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
        metric_fn=compute_f1,
        metric_name="F1",
        inst_tokens=[733, 16289, 28793],
        s_end=[733, 28748, 16289, 28793],
        suffix_is_query_len=False,
        max_tokens=32,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_ctx_len=max_ctx_len,
        stream_seed=34,
        skip_first=skip,
        fifo_max_chunks=fifo_max,
    )

    if mode == "comp":
        pair_cap = int(os.environ.get("REALISTIC_PAIR_STORE_CAP", "256"))
        prom_t = int(os.environ.get("REALISTIC_PROMOTION_THRESHOLD", "10"))
        prom_sync = os.environ.get("REALISTIC_PROMOTE_SYNC", "0") == "1"
        pair_kind = os.environ.get("REALISTIC_PAIR_STORE_KIND", "full").lower()
        delta_ratio = float(os.environ.get("REALISTIC_DELTA_TOP_K_RATIO", "0.1"))
        print(
            f"[mode=comp] pair_store_cap={pair_cap} "
            f"promotion_threshold={prom_t} promote_sync={prom_sync} "
            f"pair_store_kind={pair_kind} delta_top_k_ratio={delta_ratio}"
        )
        run_blend_eval_comp(
            **common_kwargs,
            pair_store_capacity=pair_cap,
            pair_store_kind=pair_kind,
            delta_top_k_ratio=delta_ratio,
            promotion_threshold=prom_t,
            promote_sync=prom_sync,
        )
    elif mode == "fifo":
        run_blend_eval_fifo(**common_kwargs)
    else:
        raise ValueError(f"Unknown REALISTIC_MODE={mode!r}; expected 'fifo' or 'comp'")


if __name__ == "__main__":
    main()
