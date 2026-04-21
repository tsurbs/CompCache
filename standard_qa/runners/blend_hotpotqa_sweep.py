"""Recomp-ratio sweep on HotpotQA — runs Full once, then single- and
pair-KV cached prefill at every ``recomp_ratio`` in the sweep.

Pair ratio is locked at half the single ratio (``r_pair = r_single / 2``).
"""
from utils import build_qa_prompt, compute_f1, run_blend_eval_recomp_sweep

query_prompt = (
    "\n\nAnswer the question based on the given passages."
    " Answer the question within 5 words."
    " Do NOT repeat the question or output any other words. Question: "
)

SWEEP_RATIOS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

run_blend_eval_recomp_sweep(
    dataset_path="standard_qa/inputs/hotpotqa_s.json",
    prefix_prompt=(
        "Answer the question based on the given passages."
        " Only give me the answer and do not output any other words."
        "\n\nThe following are given passages.\n"
    ),
    prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
    metric_fn=compute_f1,
    metric_name="F1",
    recomp_ratios=SWEEP_RATIOS,
    pair_ratio_fn=lambda r: r / 2.0,
    inst_tokens=[733, 16289, 28793],
    s_end=[733, 28748, 16289, 28793],
    suffix_is_query_len=False,
    max_tokens=32,
    fast_attention=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)
