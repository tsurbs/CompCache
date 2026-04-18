from __future__ import annotations

import _bootstrap_utils  # noqa: F401
from longctx_config import env_dataset, env_float, env_int, env_optional_int
from plot_longctx_benchmarks import save_nith_figures, save_nith_heatmaps
from utils import build_qa_prompt, compute_f1, run_blend_eval

query_prompt = (
    "\n\nAnswer the question directly based on the given passages."
    " Do NOT repeat the question."
    " The answer should be within 5 words.\nQuestion:"
)

_DEFAULT_DS = "longctx_eval/inputs/needle_haystack_l.json"
_DATASET = env_dataset(_DEFAULT_DS)
_MAX_CTX = env_int("LONGCTX_MAX_CTX_LEN", 8192)
_MAX_MODEL_LEN = env_optional_int("LONGCTX_MAX_MODEL_LEN")
_GPU_MEM = env_float("LONGCTX_GPU_MEMORY_UTILIZATION", 0.45)

_result = run_blend_eval(
    dataset_path=_DATASET,
    prefix_prompt=(
        "You will be asked a question after reading several passages."
        " Answer using only information in the passages."
        " Do NOT repeat the question."
        " The answer should be within 5 words.\nPassages:\n"
    ),
    prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
    metric_fn=compute_f1,
    metric_name="F1",
    inst_tokens=[733, 16289, 28793],
    s_end=[733, 28748, 16289, 28793],
    suffix_is_query_len=False,
    max_tokens=32,
    max_ctx_len=_MAX_CTX,
    max_model_len=_MAX_MODEL_LEN,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=_GPU_MEM,
)
save_nith_figures(_DATASET, _result, metric_name="F1")
save_nith_heatmaps(_DATASET, _result, metric_name="F1")
