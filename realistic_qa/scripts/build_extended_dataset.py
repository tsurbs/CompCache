
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GATEWAY = "https://ai-gateway.andrew.cmu.edu/v1"

def _repo_json(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p

def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write("\n")

def sample_rows(rows: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]

def flatten_ctx_text(example: dict) -> str:
    return "\n\n".join(c.get("text", "") for c in example.get("ctxs", []))

def ctxs_from_text_chunks(chunks: list[str]) -> list[dict]:
    return [{"title": "", "text": t} for t in chunks]

def ctxs_samsum(example: dict) -> list[dict]:
    return example.get("ctxs", [])

def build_splitter(tokenizer_id: str) -> RecursiveCharacterTextSplitter:
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    def length_function(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def split_example_context(example: dict, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    text = flatten_ctx_text(example)
    if not text.strip():
        return []
    parts = splitter.split_text(text)
    return [p for p in parts if p.strip()]

def resolve_st_device(st_device: str) -> str:
    if st_device in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return st_device

def load_st_model(model_id: str, st_device: str) -> SentenceTransformer:
    dev = resolve_st_device(st_device)
    print(f"[build] SentenceTransformer {model_id!r} on device={dev!r}")
    try:
        return SentenceTransformer(model_id, device=dev)
    except TypeError:
        m = SentenceTransformer(model_id)
        return m if dev == "cpu" else m.to(dev)

def embed_all(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    out = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(out, dtype=np.float32)

def paraphrase_three(
    client: OpenAI,
    model: str,
    question: str,
    retries: int = 4,
) -> list[str]:
    
    
    prompt = (
        "Generate exactly 3 diverse paraphrases of the question below. "
        "Keep the same answerable intent. Output ONLY a JSON array of 3 strings, "
        "no markdown or explanation.\n\nQuestion:\n"
        f"{question.strip()}"
    )
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, 
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            arr = None
            try:
                arr = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\[[\s\S]*\]", raw)
                if m:
                    arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) >= 3:
                return [str(arr[i]).strip() for i in range(3)]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Failed to paraphrase: {question[:80]}")

def build_musique_wikimqa(
    musique_path: Path,
    wikimqa_path: Path,
    out_path: Path,
    *,
    seed: int,
    n_per_set: int,
    splitter_model: str,
    st_model: str,
    gateway: str,
    api_key: str | None,
    chat_model: str,
    top_k: int,
    shuffle_seed: int,
    st_device: str = "auto",
    embed_batch_size: int = 64,
) -> None:
    mus_all = load_json(musique_path)
    wiki_all = load_json(wikimqa_path)

    mus_s = sample_rows(mus_all, n_per_set, seed)
    wiki_s = sample_rows(wiki_all, n_per_set, seed + 1)

    splitter = build_splitter(splitter_model)

    
    chunk_rows: list[tuple[str, int, list[str], str, int]] = []
    originals: list[dict] = []

    for src, data in [("musique", mus_s), ("wikimqa", wiki_s)]:
        for oi, ex in enumerate(data):
            chunks = split_example_context(ex, splitter)
            if not chunks:
                chunks = [flatten_ctx_text(ex) or " "]
            originals.append(
                {
                    "source": src,
                    "orig_idx": oi,
                    "question": ex["question"],
                    "answers": ex.get("answers", []),
                    "chunks": chunks,
                }
            )
            for ci, ct in enumerate(chunks):
                chunk_rows.append((src, oi, ex.get("answers", []), ct, ci))

    unique_texts: list[str] = []
    text_to_id: dict[str, int] = {}
    for *_, ct, _ in chunk_rows:
        if ct not in text_to_id:
            text_to_id[ct] = len(unique_texts)
            unique_texts.append(ct)

    print(f"[build] {len(originals)} originals; {len(unique_texts)} unique chunks")
    st = load_st_model(st_model, st_device)
    embs = embed_all(st, unique_texts, batch_size=embed_batch_size)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    if not api_key:
        raise SystemExit("LITELLM_KEY (or OPENAI_API_KEY) required for paraphrases")
    client = OpenAI(base_url=gateway, api_key=api_key)

    query_variants: list[tuple[dict, str, str]] = []
    for orig in originals:
        q = orig["question"]
        query_variants.append((orig, q, "original"))
        paras = paraphrase_three(client, chat_model, q)
        for pi, pq in enumerate(paras):
            query_variants.append((orig, pq, f"paraphrase_{pi + 1}"))

    st_tqdm = tqdm(query_variants, desc="retrieve")
    shuffle_rng = random.Random(shuffle_seed)
    output: list[dict] = []
    for orig, qtext, qkind in st_tqdm:
        q_emb = st.encode(
            [qtext],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        _, I = index.search(q_emb, top_k)
        idxs = list(I[0])
        chunk_texts = [unique_texts[i] for i in idxs]
        shuffle_rng.shuffle(chunk_texts)
        ex_out = {
            "question": qtext,
            "answers": list(orig["answers"]),
            "ctxs": ctxs_from_text_chunks(chunk_texts),
            "_meta": {
                "source": orig["source"],
                "orig_idx": orig["orig_idx"],
                "variant": qkind,
                "retrieved_ids": [int(i) for i in idxs],
            },
        }
        output.append(ex_out)

    save_json(out_path, output)
    print(f"[build] wrote {len(output)} rows -> {out_path}")

def build_samsum(
    samsum_path: Path,
    out_path: Path,
    *,
    seed: int,
    n_sample: int,
    st_model: str,
    gateway: str,
    api_key: str | None,
    chat_model: str,
    top_k: int,
    shuffle_seed: int,
    st_device: str = "auto",
    embed_batch_size: int = 64,
) -> None:
    data = load_json(samsum_path)
    sampled = sample_rows(data, n_sample, seed)
    chunk_rows: list[tuple[int, list[str], str]] = []
    originals: list[dict] = []

    for oi, ex in enumerate(sampled):
        ctxs = ctxs_samsum(ex)
        texts = [c.get("text", "") for c in ctxs if c.get("text")]
        if not texts:
            continue
        originals.append(
            {
                "orig_idx": oi,
                "question": ex.get("question", ex.get("input", "")),
                "answers": ex.get("answers", []),
                "chunks": texts,
            }
        )
        for ct in texts:
            chunk_rows.append((oi, ex.get("answers", []), ct))

    unique_texts: list[str] = []
    text_to_id: dict[str, int] = {}
    for _, _, ct in chunk_rows:
        if ct not in text_to_id:
            text_to_id[ct] = len(unique_texts)
            unique_texts.append(ct)

    print(f"[samsum] {len(originals)} examples; {len(unique_texts)} unique chunks")
    st = load_st_model(st_model, st_device)
    embs = embed_all(st, unique_texts, batch_size=embed_batch_size)
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    if not api_key:
        raise SystemExit("LITELLM_KEY (or OPENAI_API_KEY) required for paraphrases")
    client = OpenAI(base_url=gateway, api_key=api_key)

    query_variants: list[tuple[dict, str, str]] = []
    for orig in originals:
        q = orig["question"]
        query_variants.append((orig, q, "original"))
        paras = paraphrase_three(client, chat_model, q)
        for pi, pq in enumerate(paras):
            query_variants.append((orig, pq, f"paraphrase_{pi + 1}"))

    shuffle_rng = random.Random(shuffle_seed)
    output: list[dict] = []
    for orig, qtext, qkind in tqdm(query_variants, desc="samsum retrieve"):
        q_emb = st.encode(
            [qtext],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        _, I = index.search(q_emb, top_k)
        idxs = list(I[0])
        chunk_texts = [unique_texts[i] for i in idxs]
        shuffle_rng.shuffle(chunk_texts)
        output.append(
            {
                "question": qtext,
                "answers": list(orig["answers"]),
                "ctxs": ctxs_from_text_chunks(chunk_texts),
                "_meta": {
                    "dataset": "samsum",
                    "orig_idx": orig["orig_idx"],
                    "variant": qkind,
                    "retrieved_ids": [int(i) for i in idxs],
                },
            }
        )

    save_json(out_path, output)
    print(f"[samsum] wrote {len(output)} rows -> {out_path}")

def main() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("musique_wikimqa", "samsum", "both"),
        default="musique_wikimqa",
    )
    ap.add_argument("--musique", type=str, default="standard_qa/inputs/musique_s.json")
    ap.add_argument("--wikimqa", type=str, default="standard_qa/inputs/wikimqa_s.json")
    ap.add_argument("--samsum", type=str, default="standard_qa/inputs/samsum.json")
    ap.add_argument(
        "--out",
        type=str,
        default="realistic_qa/inputs/extended_cacheblend.json",
    )
    ap.add_argument("--out-samsum", type=str, default="realistic_qa/inputs/extended_samsum.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-per-set",
        type=int,
        default=750,
        help="750 musique + 750 wikimqa => 1500 originals, 6000 with paraphrases",
    )
    ap.add_argument("--samsum-n", type=int, default=1500)
    ap.add_argument("--splitter-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--st-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument(
        "--st-device",
        type=str,
        default=os.environ.get("REALISTIC_ST_DEVICE", "auto"),
        choices=("auto", "cuda", "cpu"),
        help="SentenceTransformer device (auto prefers CUDA when available)",
    )
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=0,
        metavar="N",
        help="ST encode batch size (0 = 128 if cuda else 64)",
    )
    ap.add_argument("--gateway", type=str, default=os.environ.get("LITELLM_BASE_URL", _DEFAULT_GATEWAY))
    ap.add_argument(
        "--chat-model",
        type=str,
        default=os.environ.get("LITELLM_CHAT_MODEL", "gpt-4.1-mini-2025-04-14"),
    )
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--shuffle-seed", type=int, default=34)
    args = ap.parse_args()

    gw = args.gateway.rstrip("/")
    if not gw.endswith("v1"):
        gw = f"{gw}/v1"
    args.gateway = gw

    api_key = os.environ.get("LITELLM_KEY") or os.environ.get("OPENAI_API_KEY")

    ebs = args.embed_batch_size
    if ebs <= 0:
        ebs = 128 if resolve_st_device(args.st_device) == "cuda" else 64

    if args.mode in ("musique_wikimqa", "both"):
        build_musique_wikimqa(
            _repo_json(args.musique),
            _repo_json(args.wikimqa),
            _repo_json(args.out),
            seed=args.seed,
            n_per_set=args.n_per_set,
            splitter_model=args.splitter_model,
            st_model=args.st_model,
            gateway=args.gateway.rstrip("/"),
            api_key=api_key,
            chat_model=args.chat_model,
            top_k=args.top_k,
            shuffle_seed=args.shuffle_seed,
            st_device=args.st_device,
            embed_batch_size=ebs,
        )

    if args.mode in ("samsum", "both"):
        build_samsum(
            _repo_json(args.samsum),
            _repo_json(args.out_samsum),
            seed=args.seed,
            n_sample=args.samsum_n,
            st_model=args.st_model,
            gateway=args.gateway.rstrip("/"),
            api_key=api_key,
            chat_model=args.chat_model,
            top_k=args.top_k,
            shuffle_seed=args.shuffle_seed,
            st_device=args.st_device,
            embed_batch_size=ebs,
        )

if __name__ == "__main__":
    main()
