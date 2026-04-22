"""CompCache on SAMSum using the sparse-delta pair store.

Mirrors ``blend_samsum_comp.py`` (Full-Joint pair store) one-to-one so
ROUGE-L and TTFT are directly comparable.  Artifacts land under
``*_comp_delta_*`` so they coexist with the baseline's ``*_comp_*``.

Env overrides: ``STANDARD_COMP_DELTA_TOP_K_RATIO``,
``STANDARD_COMP_ARTIFACT_SUFFIX`` (see ``utils.run_blend_eval_comp``).
"""

from utils import build_fewshot_prompt, compute_rl, run_blend_eval_comp

run_blend_eval_comp(
    dataset_path="standard_qa/inputs/samsum.json",
    prefix_prompt=(
        "Summarize the dialogue into a few short sentences."
        " The following are some examples.\n\n"
    ),
    prompt_builder=build_fewshot_prompt,
    metric_fn=lambda pred, ans, tok: compute_rl(pred, ans),
    metric_name="rl",
    suffix_is_query_len=True,
    max_ctx_len=3400,
    max_tokens=128,
    recomp_ratio=0.5,
    fast_attention=True,
    extra_metadata={"attn_bias": None},
    post_process=lambda s: s.lstrip('\n').split('\n')[0],
    clear_hack_kv=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    pair_store_kind="delta",
    delta_top_k_ratio=0.10,
    artifact_suffix="comp_delta",
)
