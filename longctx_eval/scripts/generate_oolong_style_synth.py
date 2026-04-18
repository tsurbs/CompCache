"""Generate synthetic multi-chunk aggregation JSON (Oolong-style counting / coverage).

Usage::

  python longctx_eval/scripts/generate_oolong_style_synth.py -n 250 --seed 7 \\
    --out longctx_eval/inputs/oolong_l.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pad(rng: random.Random, lines: list[str], target_chars: int) -> str:
    """Extend with fluff but never truncate ``lines`` (counts must stay valid)."""
    base = "\n".join(lines)
    fluff = [
        "Ops note: ignore rows tagged TEST.",
        "Reminder: SLA clock pauses on vendor holidays.",
        "Watermark: internal distribution only.",
        "Cross-ref: see appendix C for definitions.",
    ]
    while len(base) < target_chars:
        base += "\n" + rng.choice(fluff)
    return base


def _make_example(rng: random.Random) -> dict:
    n_chunks = rng.randint(2, 5)
    # Items are "ITEM k STATUS" spread across chunks; count STATUS==shipped
    total_items = rng.randint(8, 24)
    shipped_mask = [rng.random() < 0.45 for _ in range(total_items)]
    n_shipped = sum(shipped_mask)

    per = [[] for _ in range(n_chunks)]
    for k in range(total_items):
        per[rng.randrange(n_chunks)].append((k, shipped_mask[k]))

    statuses = ("shipped", "pending", "cancelled")
    lines_by_chunk: list[list[str]] = [[] for _ in range(n_chunks)]
    for ci, bucket in enumerate(per):
        for k, is_ship in bucket:
            st = "shipped" if is_ship else rng.choice(("pending", "cancelled"))
            lines_by_chunk[ci].append(f"ORD-{1000+k}: status {st} line {ci}")

    target = rng.randint(400, 2200)
    ctxs = []
    for ci in range(n_chunks):
        body = _pad(rng, lines_by_chunk[ci], target // n_chunks + rng.randint(50, 400))
        ctxs.append({"title": f"c{ci}", "text": body})

    q = "Across all passages, how many orders have status shipped?"
    return {
        "question": q,
        "answers": [str(n_shipped)],
        "ctxs": ctxs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num", type=int, default=220)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rng = random.Random(args.seed)
    root = _repo_root()
    out = args.out or (root / "longctx_eval" / "inputs" / "oolong_l.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [_make_example(rng) for _ in range(args.num)]
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
