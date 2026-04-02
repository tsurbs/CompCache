import json
import collections
import string
import re
import hashlib
import math
from itertools import combinations
from pathlib import Path
from rouge_score import rouge_scorer


def save_ttft_histogram(
    dataset_path: str,
    ttft_blend: list[float],
    ttft_full: list[float],
    *,
    cached_label: str = "Cached (CacheBlend)",
    full_label: str = "Full prefill",
    extra_metadata: dict | None = None,
) -> tuple[str, str | None]:
    """Persist TTFT samples and an overlaid histogram (cached vs full prefill) next to the dataset.

    TTFT values are seconds (vLLM ``first_token_time - first_scheduled_time``).
    Writes ``{stem}_ttft_hist.json`` and ``{stem}_ttft_hist.png`` (PNG omitted if matplotlib
    is missing).
    """
    n = len(ttft_blend)
    if n == 0 or len(ttft_full) != n:
        return "", None

    base = Path(dataset_path).resolve()
    json_path = base.with_name(f"{base.stem}_ttft_hist.json")
    png_path = base.with_name(f"{base.stem}_ttft_hist.png")

    meta = dict(extra_metadata or {})
    meta["dataset"] = str(base)
    meta["n_queries"] = n
    payload = {
        "ttft_blend_seconds": [float(x) for x in ttft_blend],
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "metadata": meta,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    png_written: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        b_ms = np.asarray(ttft_blend, dtype=float) * 1e3
        f_ms = np.asarray(ttft_full, dtype=float) * 1e3
        edges = np.histogram_bin_edges(
            np.concatenate([b_ms, f_ms]), bins="auto"
        )
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.hist(
            b_ms,
            bins=edges,
            alpha=0.55,
            label=cached_label,
            color="#1f77b4",
            density=True,
        )
        ax.hist(
            f_ms,
            bins=edges,
            alpha=0.55,
            label=full_label,
            color="#ff7f0e",
            density=True,
        )
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("Density")
        ax.set_title("TTFT: cached vs full prefill")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        png_written = str(png_path)
    except ImportError:
        pass

    if png_written:
        print(f"TTFT histogram: {png_written}")
    print(f"TTFT histogram data: {json_path}")
    return str(json_path), png_written

def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s


def _coerce_answer_text(a) -> str:
    """Gold answers may be str, nested list (dataset quirks), or None."""
    if isinstance(a, str):
        return a
    if a is None:
        return ""
    if isinstance(a, list):
        if not a:
            return ""
        if all(isinstance(x, str) for x in a):
            return a[0]
        return _coerce_answer_text(a[0])
    return str(a)


def normalize_answer(s):
    s = _coerce_answer_text(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def build_qa_prompt(example, query_prompt):

    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    #ex_prompt = f"{docs_text}\n\nBased on these texts, answer the question:\nQ: {q}\nA:"
    #q_prompt = f"\n\nAnswer the question based on the given passages. Answer the question within 5 words. Do NOT repeat the question or output any other words. Question: {q}\nAnswer:"
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt

def build_fewshot_prompt(example):
    q = "\n\n"+example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL


def get_doc_ids(example):
    """Extract stable document identifiers from a dataset example.

    Uses 'title' when non-empty, otherwise a truncated SHA-256 of the text.
    """
    ids = []
    for i, ctx in enumerate(example.get("ctxs", [])):
        if ctx.get("title"):
            ids.append(ctx["title"])
        else:
            text = ctx.get("text", f"chunk_{i}")
            ids.append(hashlib.sha256(text.encode()).hexdigest()[:16])
    return ids


class CoRetrievalTracker:
    """Tracks pairwise document co-retrieval frequency across queries
    and fits a Zipf power-law model to the resulting distribution."""

    def __init__(self):
        self.pair_counts = collections.Counter()
        self.doc_counts = collections.Counter()
        self.retrieval_sets = []

    def record(self, doc_ids):
        """Record one retrieval set (the list of doc ids returned for a query)."""
        self.retrieval_sets.append(list(doc_ids))
        for doc_id in doc_ids:
            self.doc_counts[doc_id] += 1
        for pair in combinations(sorted(set(doc_ids)), 2):
            self.pair_counts[pair] += 1

    def fit_zipf(self):
        """Estimate Zipf exponent alpha via log-log OLS on (rank, frequency).

        Returns (alpha, r_squared) or (None, None) if insufficient data.
        """
        ranked = self.pair_counts.most_common()
        if len(ranked) < 2:
            return None, None

        log_r = [math.log(i + 1) for i in range(len(ranked))]
        log_f = [math.log(freq) for _, freq in ranked]
        n = len(log_r)
        mx = sum(log_r) / n
        my = sum(log_f) / n
        ss_xx = sum((x - mx) ** 2 for x in log_r)
        ss_xy = sum((x - mx) * (y - my) for x, y in zip(log_r, log_f))
        ss_yy = sum((y - my) ** 2 for y in log_f)
        if ss_xx == 0 or ss_yy == 0:
            return None, None

        alpha = -ss_xy / ss_xx
        r_sq = (ss_xy ** 2) / (ss_xx * ss_yy)
        return alpha, r_sq

    def summary(self):
        """Print and return co-retrieval frequency analysis."""
        ranked = self.pair_counts.most_common()
        alpha, r_sq = self.fit_zipf()

        print("\n========== Co-Retrieval Frequency Analysis ==========")
        print(f"Queries processed:        {len(self.retrieval_sets)}")
        print(f"Unique documents:         {len(self.doc_counts)}")
        print(f"Unique co-retrieved pairs: {len(self.pair_counts)}")

        if ranked:
            freqs = [f for _, f in ranked]
            total = sum(freqs)
            print(f"Total co-retrieval events: {total}")

            print("\nTop 10 co-retrieved pairs:")
            for i, (pair, freq) in enumerate(ranked[:10]):
                print(f"  {i+1:>2}. {pair}: {freq}  ({freq/total*100:.1f}%)")

            if len(freqs) >= 5:
                top_20_idx = max(1, len(freqs) // 5)
                top_20_sum = sum(freqs[:top_20_idx])
                print(
                    f"\nConcentration: top 20% of pairs ({top_20_idx}/{len(freqs)}) "
                    f"cover {top_20_sum/total*100:.1f}% of co-retrievals"
                )

        if alpha is not None:
            print(f"\nZipf fit: α = {alpha:.3f}, R² = {r_sq:.3f}")
            if alpha > 0.8:
                print("  -> Strong power-law: hierarchical KV caching would be highly effective")
            elif alpha > 0.4:
                print("  -> Moderate power-law: pair-aware caching offers some benefit")
            else:
                print("  -> Weak power-law: co-retrieval is relatively uniform")

        return {
            "alpha": alpha,
            "r_squared": r_sq,
            "n_queries": len(self.retrieval_sets),
            "n_unique_docs": len(self.doc_counts),
            "n_unique_pairs": len(self.pair_counts),
            "top_pairs": [(list(pair), freq) for pair, freq in ranked[:50]],
            "doc_frequencies": dict(self.doc_counts.most_common()),
        }


def run_blend_eval(
    dataset_path,
    prefix_prompt,
    prompt_builder,
    metric_fn,
    metric_name,
    inst_tokens=None,
    s_end=None,
    suffix_is_query_len=True,
    max_ctx_len=None,
    max_tokens=32,
    recomp_ratio=None,
    fast_attention=None,
    extra_metadata=None,
    post_process=None,
    clear_hack_kv=False,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=0.5,
    num_layers=32,
):
    """Run CacheBlend evaluation with co-retrieval frequency tracking.

    Args:
        dataset_path: Path to the JSON dataset.
        prefix_prompt: System/instruction prefix text.
        prompt_builder: callable(example) -> (doc_prompts, q_prompt).
        metric_fn: callable(prediction, answer, tokenizer) -> float.
        metric_name: Display name for the metric (e.g. "F1", "rl").
        inst_tokens: Optional instruction token ids prepended before the
                     encoded prefix_prompt (e.g. [INST] tokens).
        s_end: End-of-sequence token ids appended after the query (default []).
        suffix_is_query_len: If True, suffix_len = len(q_ids + s_end).
                             If False, suffix_len = 1.
        max_ctx_len: Optional max context length; chunks are dropped from the
                     middle when exceeded.  None disables.
        max_tokens: Max new tokens to generate.
        recomp_ratio: Selective recomputation ratio (None = don't set).
        fast_attention: Fast attention flag (None = don't set).
        extra_metadata: Dict of extra cache_fuse_metadata keys to set at the
                        start of each example (e.g. {"attn_bias": None}).
        post_process: Optional callable(str) -> str applied to generation text.
        clear_hack_kv: Set hack_kv to None after reading each layer.
        model_name: HuggingFace model identifier.
        gpu_memory_utilization: vLLM GPU memory fraction.
        num_layers: Number of transformer layers.

    Returns:
        Dict with TTFT lists, metric lists, and co-retrieval statistics.
    """
    import numpy as np
    import torch
    from itertools import chain
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    eval_dataset = load_dataset(dataset_path)

    llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm.set_tokenizer(tokenizer)

    if s_end is None:
        s_end = []
    if extra_metadata is None:
        extra_metadata = {}

    prefix_ids = tokenizer.encode(prefix_prompt)[1:]
    s_start_full = list(inst_tokens) + prefix_ids if inst_tokens else prefix_ids
    s_start_len = len(s_start_full) + 1

    s_start = []
    s_start_1_len = len(s_start) + 1

    tracker = CoRetrievalTracker()
    ttft_blend = []
    ttft_full = []
    metric_blend = []
    metric_full = []

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

        model_ref = (
            llm.llm_engine.model_executor.driver_worker
            .model_runner.model.model
        )
        cache_fuse_metadata = model_ref.cache_fuse_metadata
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False
        for k, v in extra_metadata.items():
            cache_fuse_metadata[k] = v

        doc_chunk_ids = [s_start + chunk_ids for chunk_ids in doc_chunk_ids]
        doc_chunk_ids = [s_start_full] + doc_chunk_ids
        doc_chunk_ids = doc_chunk_ids + [s_start + q_ids + s_end]

        last_len = len(q_ids + s_end) if suffix_is_query_len else len([q_ids + s_end])

        # --- KV collection phase ---
        cache_fuse_metadata['collect'] = True
        cache_fuse_metadata['check'] = False
        chunk_past_key_values = []

        for i in range(len(doc_chunk_ids)):
            prompts = [tokenizer.decode(doc_chunk_ids[i])]
            llm.generate(prompts, sampling_params)

            for j in range(num_layers):
                past_key_values = model_ref.layers[j].self_attn.hack_kv
                if i == 0:
                    temp_k = past_key_values[0][:s_start_len].clone()
                    temp_v = past_key_values[1][:s_start_len].clone()
                else:
                    temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
                    temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()

                if i == 0:
                    chunk_past_key_values.append([temp_k, temp_v])
                else:
                    chunk_past_key_values[j][0] = torch.cat(
                        (chunk_past_key_values[j][0], temp_k), dim=0
                    )
                    chunk_past_key_values[j][1] = torch.cat(
                        (chunk_past_key_values[j][1], temp_v), dim=0
                    )

                if clear_hack_kv:
                    model_ref.layers[j].self_attn.hack_kv = None

            model_ref.old_kvs = chunk_past_key_values

        # --- Build full input ---
        input_ids = []
        for i in range(len(doc_chunk_ids)):
            if i == 0:
                temp_ids = doc_chunk_ids[i]
            else:
                temp_ids = doc_chunk_ids[i][s_start_1_len - 1:]
            input_ids += temp_ids
        input_prompt = tokenizer.decode(input_ids)

        # --- Cached generation ---
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata['check'] = True
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['suffix_len'] = last_len
        if recomp_ratio is not None:
            cache_fuse_metadata['recomp_ratio'] = recomp_ratio
        if fast_attention is not None:
            cache_fuse_metadata['fast_attention'] = fast_attention

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

        # --- Normal generation ---
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        cache_fuse_metadata['check'] = False
        cache_fuse_metadata['collect'] = False
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

    # --- Summary ---
    print("\n=============== Result Summary =====================")
    print(f"TTFT with cache: {np.mean(ttft_blend)}")
    print(f"TTFT with full prefill: {np.mean(ttft_full)}")
    print(f"{metric_name} with cache: {np.mean(metric_blend)}")
    print(f"{metric_name} with full prefill: {np.mean(metric_full)}")

    coret_stats = tracker.summary()

    output_path = dataset_path.replace('.json', '_coretrieval.json')
    with open(output_path, 'w') as f:
        json.dump(coret_stats, f, indent=2, default=str)
    print(f"\nCo-retrieval data saved to {output_path}")

    save_ttft_histogram(dataset_path, ttft_blend, ttft_full)

    return {
        "ttft_blend": ttft_blend,
        "ttft_full": ttft_full,
        f"{metric_name}_blend": metric_blend,
        f"{metric_name}_full": metric_full,
        "coretrieval": coret_stats,
    }