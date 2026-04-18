"""Build ``longctx_eval/inputs/oolong_s.json`` from oolongbench/oolong-synth.

Requires: ``pip install datasets``

Uses streaming so the full HF parquet is never loaded into memory. Example::

  python longctx_eval/scripts/prepare_oolong_blend.py --max-samples 80 \\
    --per-example-chars 20000 --max-chunk-chars 3500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _chunk_paragraphs(text: str, max_chunk_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for p in parts:
        if len(p) <= max_chunk_chars:
            chunks.append(p)
            continue
        for i in range(0, len(p), max_chunk_chars):
            chunks.append(p[i : i + max_chunk_chars])
    if not chunks:
        chunks = [text[:max_chunk_chars]]
    return chunks


def _trim_ctxs(ctxs: list[dict], max_total_chars: int) -> list[dict]:
    total = 0
    out: list[dict] = []
    for c in ctxs:
        t = c["text"]
        if total >= max_total_chars:
            break
        if total + len(t) > max_total_chars:
            t = t[: max_total_chars - total]
        if not t:
            break
        out.append({"title": c.get("title") or "", "text": t})
        total += len(t)
    return out


def row_to_example(
    row: dict,
    *,
    max_chunk_chars: int,
    max_total_chars: int,
) -> dict | None:
    ctx_text = row.get("context_window_text") or ""
    q = row.get("question")
    ans = row.get("answer")
    if q is None or ans is None:
        return None
    pieces = _chunk_paragraphs(ctx_text, max_chunk_chars)
    ctxs = [{"title": f"c{i}", "text": p} for i, p in enumerate(pieces)]
    ctxs = _trim_ctxs(ctxs, max_total_chars)
    if not ctxs:
        return None
    return {
        "question": str(q).strip(),
        "ctxs": ctxs,
        "answers": [str(ans).strip()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--per-example-chars", type=int, default=24_000)
    parser.add_argument("--max-chunk-chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: longctx_eval/inputs/oolong_s.json)",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    root = _repo_root()
    out_path = args.out or (root / "longctx_eval" / "inputs" / "oolong_s.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("oolongbench/oolong-synth", split="train", streaming=True)
    records: list[dict] = []
    skipped = 0
    for row in ds:
        if len(records) >= args.max_samples:
            break
        ex = row_to_example(
            row,
            max_chunk_chars=args.max_chunk_chars,
            max_total_chars=args.per_example_chars,
        )
        if ex is None:
            skipped += 1
            continue
        records.append(ex)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} examples (skipped {skipped}) -> {out_path}")


if __name__ == "__main__":
    main()
