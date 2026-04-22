"""CompCache on WikiMQA using the sparse-delta pair store.

Mirrors ``blend_wikimqa_comp.py`` (Full-Joint pair store) one-to-one so
F1 and TTFT are directly comparable.  Artifacts land under
``*_comp_delta_*`` so they coexist with the baseline's ``*_comp_*``.

Env overrides: ``STANDARD_COMP_DELTA_TOP_K_RATIO``,
``STANDARD_COMP_ARTIFACT_SUFFIX`` (see ``utils.run_blend_eval_comp``).
"""

from utils import build_qa_prompt, compute_f1, run_blend_eval_comp

query_prompt = (
    "\n\nAnswer the question based on the given passages."
    " Answer the question within 5 words."
    " Do NOT repeat the question or output any other words. Question: "
)

run_blend_eval_comp(
    dataset_path="standard_qa/inputs/wikimqa_s.json",
    prefix_prompt=(
        "Answer the question based on the given passages."
        " Only give me the answer and do not output any other words."
        "\n\nThe following are given passages.\n"
    ),
    prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
    metric_fn=lambda pred, ans, tok: compute_f1(pred, ans[0], tok),
    metric_name="F1",
    inst_tokens=[733, 16289, 28793],
    s_end=[733, 28748, 16289, 28793],
    suffix_is_query_len=False,
    max_tokens=32,
    recomp_ratio=0.5,
    fast_attention=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    pair_store_kind="delta",
    delta_top_k_ratio=0.10,
    artifact_suffix="comp_delta",
)
