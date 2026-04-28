# Realistic / Extended QA

This is based on the following section from the CacheBlend paper: 
> We split contexts into 512-token chunks with Langchain and use the original 200-400 token chunks in 
SAMSum. We also create a synthetic dataset to simulate the chunk reuse in RAG scenarios. Specifically, we 
randomly pick 1500 queries in the Musique and 2WikiMQA datasets each and build a context chunk database by 
splitting each query’s context into 512-token chunks [29] with Langchain [5]. For each query, we use GPT4 
API to generate 3 more similar queries. In the 6000 queries (1500 original + 4500 simulated), we retrieve 
the top-6 chunks8 based on L2 distance, in a random order[34]. We refer to these datasets as Musique 
extended and 2WikiMQA extended. We only report for baselines with similar quality and skip the result for 
the first 1K queries as the initial storage is completely empty.
Breaking this down, we'll need the following components to mimic the extended dataset feature:
1. Split contexts into 512-token chunks with Langchain and use the original 200-400 token chunks in SAMSum
2. Create a synthetic dataset to simulate the chunk reuse in RAG scenarios. 
   1. Specifically, we randomly pick 1500 queries in the Musique and 2WikiMQA datasets each at random and 
   2. Build a context chunk (faiss) database by splitting each query’s context into 512-token chunks [29] 
   with Langchain [5].
   3. For each query, use GPT4 API to generate 3 more similar queries
   4. In the 6000 queries (1500 original + 4500 simulated), retrieve the top-6 chunks based on L2 
   distance, in a random order[34], embedded with SentenceTransformers
3. Create a special runner that dynamically caches as you would seeing these queries come in in real time
   1. Feed queries + precomputed context in random (seed 34) order
   2. Cache as much as possible per the CacheBlend protocol, evict with FIFO
5. Create the modal hookup in `modal_runner.py`

## CompCache (composition-aware) mode

Same runner as FIFO, but with pair promotion and joint KV storage (`run_blend_eval_comp` in `realistic_qa/runners/blend_realistic.py`).

Example (tiny dataset, after installing realistic QA + vLLM deps):

```bash
export REALISTIC_MODE=comp
export REALISTIC_DATASET=realistic_qa/inputs/extended_tiny.json
python realistic_qa/runners/blend_realistic.py
```

Outputs next to the dataset include `{stem}_comp_scores.json`, `{stem}_comp_coretrieval.json`, `{stem}_comp_ttft_warmup.json`, and `{stem}_comp_ttft_hist.json`. Optional env: `REALISTIC_PAIR_STORE_CAP`, `REALISTIC_PROMOTION_THRESHOLD`, `REALISTIC_PROMOTE_SYNC=1`, `REALISTIC_TTFT_ROLL_WINDOW`.