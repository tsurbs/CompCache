from utils import build_qa_prompt, compute_f1, run_blend_eval

query_prompt = (
    "\n\nAnswer the question directly based on the given passages."
    " Do NOT repeat the question."
    " The answer should be within 5 words. \nQuestion:"
)

run_blend_eval(
    dataset_path="standard_qa/inputs/multihop_rag_s.json",
    prefix_prompt=(
        "You will be asked a question after reading several passages."
        " Please directly answer the question based on the given passages."
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
    recomp_ratio=0.5,
    fast_attention=True,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
)
