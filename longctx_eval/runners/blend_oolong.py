from __future__ import annotations

import _bootstrap_utils  # noqa: F401
from longctx_config import env_dataset, env_float, env_int, env_optional_int
from plot_longctx_benchmarks import save_oolong_style_figures
from utils import build_qa_prompt, compute_f1, run_blend_eval

query_prompt = (
    "\n\nAnswer the question directly using only the given passages."
    " Be concise. Do NOT repeat the question.\nQuestion:"
)

_DEFAULT_DS = "longctx_eval/inputs/oolong_l.json"
_DATASET = env_dataset(_DEFAULT_DS)
_MAX_CTX = env_int("LONGCTX_MAX_CTX_LEN", 8192)
_MAX_MODEL_LEN = env_optional_int("LONGCTX_MAX_MODEL_LEN")
_GPU_MEM = env_float("LONGCTX_GPU_MEMORY_UTILIZATION", 0.45)

_result = run_blend_eval(
    dataset_path=_DATASET,
    prefix_prompt=(
        "You are given several text chunks drawn from a longer context."
        " Read them and answer the question; you may need to combine facts across chunks.\n"
        "Passages:\n"
    ),
    prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
    metric_fn=compute_f1,
    metric_name="F1",
    inst_tokens=[733, 16289, 28793],
    s_end=[733, 28748, 16289, 28793],
    suffix_is_query_len=False,
    max_tokens=48,
    max_ctx_len=_MAX_CTX,
    max_model_len=_MAX_MODEL_LEN,
    recomp_ratio=0.18,
    fast_attention=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=_GPU_MEM,
)
save_oolong_style_figures(_DATASET, _result, metric_name="F1")
