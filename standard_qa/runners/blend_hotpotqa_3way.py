from utils import build_qa_prompt, compute_f1, run_blend_eval_three_way

query_prompt = (
    "\n\nAnswer the question based on the given passages."
    " Answer the question within 5 words."
    " Do NOT repeat the question or output any other words. Question: "
)

run_blend_eval_three_way(
    dataset_path="standard_qa/inputs/hotpotqa_s.json",
    prefix_prompt=(
        "Answer the question based on the given passages."
        " Only give me the answer and do not output any other words."
        "\n\nThe following are given passages.\n"
    ),
    prompt_builder=lambda ex: build_qa_prompt(ex, query_prompt),
    metric_fn=compute_f1,
    metric_name="F1",
    inst_tokens=[733, 16289, 28793],
    s_end=[733, 28748, 16289, 28793],
    suffix_is_query_len=False,
    max_tokens=32,
    recomp_ratio=0.5,
    pair_recomp_ratio=0.25,
    fast_attention=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)
