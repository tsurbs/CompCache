from utils import build_fewshot_prompt, compute_rl, run_blend_eval

run_blend_eval(
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
    recomp_ratio=0.18,
    fast_attention=True,
    extra_metadata={"attn_bias": None},
    post_process=lambda s: s.lstrip('\n').split('\n')[0],
    clear_hack_kv=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)
