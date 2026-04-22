"""CompCache on MultiHop-RAG using the sparse-delta pair store.

Mirrors ``blend_multihop_rag_comp.py`` (Full-Joint pair store) one-to-one so
F1 and TTFT are directly comparable, differing only in the pair KV store:
``SparseDeltaPairStore`` keeps just ``Δ = joint_KV − cat(ind_a, ind_b)``
sparsified to the top-``top_k_ratio`` positions per layer.  Artifacts land
under ``*_comp_delta_*`` so they coexist with the baseline's ``*_comp_*``
run on the same dataset.

Env overrides:

- ``STANDARD_COMP_DELTA_TOP_K_RATIO`` — delta sparsity (default ``0.10``).
- ``STANDARD_COMP_ARTIFACT_SUFFIX`` — artifact filename suffix
  (default ``"comp_delta"``).
"""

from utils import build_qa_prompt, compute_f1, run_blend_eval_comp

query_prompt = (
    "\n\nAnswer the question directly based on the given passages."
    " Do NOT repeat the question."
    " The answer should be within 5 words. \nQuestion:"
)

run_blend_eval_comp(
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
    pair_store_kind="delta",
    delta_top_k_ratio=0.10,
    artifact_suffix="comp_delta",
)
