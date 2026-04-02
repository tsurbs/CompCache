"""Build `standard_qa/inputs/hotpotqa_s.json` and `multihop_rag_s.json` for blend eval.

Requires: pip install datasets

Usage (from repo root):
  .venv_hf/bin/python standard_qa/runners/prepare_blend_hf_datasets.py
  .venv_hf/bin/python standard_qa/runners/prepare_blend_hf_datasets.py --max-samples 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def hotpot_to_blend(row: dict) -> dict:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    ctxs = []
    for title, sents in zip(titles, sentences):
        text = " ".join(sents) if isinstance(sents, (list, tuple)) else str(sents)
        ctxs.append({"title": title if title is not None else "", "text": text})
    return {
        "question": row["question"],
        "ctxs": ctxs,
        "answers": [row["answer"]],
    }


def multihop_rag_to_blend(row: dict) -> dict:
    ctxs = []
    for ev in row.get("evidence_list") or []:
        if not isinstance(ev, dict):
            continue
        title = ev.get("title") or ev.get("source") or ""
        fact = ev.get("fact") or ""
        ctxs.append({"title": str(title) if title else "", "text": str(fact)})
    return {
        "question": row["query"],
        "ctxs": ctxs,
        "answers": [row["answer"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Cap rows per dataset (default 200, matching wikimqa_s scale).",
    )
    parser.add_argument(
        "--hotpot-split",
        default="validation",
        choices=("train", "validation", "test"),
        help="HotpotQA fullwiki split to draw from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: standard_qa/inputs/).",
    )
    args = parser.parse_args()

    root = _repo_root()
    out_dir = args.out_dir or (root / "standard_qa" / "inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = args.max_samples if args.max_samples > 0 else None

    print("Loading HotpotQA (fullwiki)...")
    hp = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=args.hotpot_split)
    if cap is not None:
        hp = hp.select(range(min(cap, len(hp))))
    hotpot_records = [hotpot_to_blend(hp[i]) for i in range(len(hp))]
    hotpot_path = out_dir / "hotpotqa_s.json"
    with open(hotpot_path, "w", encoding="utf-8") as f:
        json.dump(hotpot_records, f, ensure_ascii=False)
    print(f"Wrote {len(hotpot_records)} examples -> {hotpot_path}")

    print("Loading MultiHopRAG...")
    mhr = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
    if cap is not None:
        mhr = mhr.select(range(min(cap, len(mhr))))
    mhr_records = [multihop_rag_to_blend(mhr[i]) for i in range(len(mhr))]
    mhr_path = out_dir / "multihop_rag_s.json"
    with open(mhr_path, "w", encoding="utf-8") as f:
        json.dump(mhr_records, f, ensure_ascii=False)
    print(f"Wrote {len(mhr_records)} examples -> {mhr_path}")


if __name__ == "__main__":
    main()
