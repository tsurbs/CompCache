"""Blend evaluation on a Zipf-distributed version of WikiMQA.

Generates a synthetic dataset where the query at rank r is repeated
floor(C/r) times, then runs the full CacheBlend evaluation.  The
co-retrieval tracker inside run_blend_eval will see realistic
power-law query popularity and produce a meaningful Zipf fit.
"""

import json
from utils import build_qa_prompt, compute_f1, load_dataset, run_blend_eval

SOURCE_PATH = "standard_qa/inputs/wikimqa_s.json"
ZIPF_PATH = "standard_qa/inputs/wikimqa_zipf.json"
C = 100


def generate_zipf_dataset():
    source = load_dataset(SOURCE_PATH)
    expanded = []
    for rank, ex in enumerate(source, start=1):
        freq = max(1, C // rank)
        expanded.extend([ex] * freq)
    with open(ZIPF_PATH, "w") as f:
        json.dump(expanded, f)
    print(f"Generated {ZIPF_PATH}: {len(expanded)} examples "
          f"from {len(source)} originals (C={C})")
    return expanded


generate_zipf_dataset()

query_prompt = (
    "\n\nAnswer the question based on the given passages."
    " Answer the question within 5 words."
    " Do NOT repeat the question or output any other words. Question: "
)

run_blend_eval(
    dataset_path=ZIPF_PATH,
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
)
