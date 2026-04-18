"""Generate needle-in-a-haystack JSON for CacheBlend.

**Grid mode (default)** fills a dense depth × context lattice so NITH heatmaps are not sparse.
Context length targets scale from ``--min-chars`` to ``~max_token_budget * chars_per_token``.

Usage::

  # Dense 10×10 × 2 replicas = 200 examples, ~8k-token-scale context (default)
  python longctx_eval/scripts/generate_needle_haystack.py --out longctx_eval/inputs/needle_haystack_l.json

  # Match a longer runner cap (e.g. LONGCTX_MAX_CTX_LEN=16384)
  python longctx_eval/scripts/generate_needle_haystack.py --max-token-budget 16384 \\
    --depth-steps 12 --context-steps 12 --replicas 2

  # Legacy random mode (sparse heatmaps)
  python longctx_eval/scripts/generate_needle_haystack.py --random -n 250 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _filler_sentence(rng: random.Random, tag: str) -> str:
    parts = [
        f"{tag}: routing ticket batch {rng.randint(1000, 9999)} to the backlog.",
        "Quarterly compliance slides were filed without incident.",
        "Escalations after 18:00 go to the regional duty pager.",
        "Do not paste customer PII into the shared scratchpad.",
        "Badge revalidation windows are announced in the lobby kiosk.",
        "VPN split-tunnel policy remains unchanged this sprint.",
        "Coffee budget line moved under facilities, not engineering.",
        "Parking deck B will close for pressure-wash on Saturday.",
        "Latency SLO discussions belong in the weekly infra forum.",
        "Archive exports must retain checksum manifests per policy 7B.",
    ]
    return rng.choice(parts)


def _filler_block(rng: random.Random, tag: str, min_words: int = 30) -> str:
    out: list[str] = []
    while len(" ".join(out)) < min_words * 5:
        out.append(_filler_sentence(rng, tag))
    return " ".join(out)


def _pick_qa(rng: random.Random) -> tuple[str, str, str]:
    kinds = [
        ("code", lambda: f"SIG-{rng.randint(100, 999)}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}", "What is the signal code in the classified line?"),
        ("city", lambda: rng.choice(["Bristol", "Tucson", "Osaka", "Lille", "Curitiba", "Gdansk", "Nantes", "Valencia"]), "Which city is named as the pilot site?"),
        ("num", lambda: str(rng.randint(2, 9)), "How many spare nodes were approved for the rack?"),
        ("pin", lambda: str(rng.randint(1000, 9999)), "What is the maintenance PIN in the note?"),
        ("initials", lambda: rng.choice(["K.J. Ng", "M. Ortega", "R. Patel", "S. Lind", "T. Okonkwo"]), "Whose initials appear on the waiver?"),
        ("ratio", lambda: f"{rng.randint(30, 70)}/{100 - rng.randint(30, 70)} mix", "What mixture ratio is locked for the coolant loop?"),
        ("sku", lambda: f"SKU-{rng.randint(1000, 9999)}", "Which SKU is flagged for recall in the memo?"),
    ]
    _, val_fn, q_template = rng.choice(kinds)
    return val_fn(), q_template


def _make_grid_example(
    rng: random.Random,
    *,
    depth_bin: int,
    context_bin: int,
    depth_steps: int,
    context_steps: int,
    min_chars: int,
    max_chars: int,
    n_ctx: int,
) -> dict:
    """One example targeting the center of (depth_bin, context_bin)."""
    d_lo = depth_bin * 100.0 / depth_steps
    d_hi = (depth_bin + 1) * 100.0 / depth_steps
    target_depth_pct = 0.5 * (d_lo + d_hi)

    c_lo = min_chars + context_bin * (max_chars - min_chars) / context_steps
    c_hi = min_chars + (context_bin + 1) * (max_chars - min_chars) / context_steps
    target_chars = int(0.5 * (c_lo + c_hi))

    needle_chunk = int(round((target_depth_pct / 100.0) * max(1, n_ctx - 1)))
    needle_chunk = min(max(needle_chunk, 0), n_ctx - 1)

    secret, q_template = _pick_qa(rng)
    titles = [
        "Ops bulletin",
        "Legal scratch pad",
        "Runbook fragment",
        "Vendor chatter",
        "HR reminder",
        "Facilities log",
        "Engineering note",
        "Random standup",
    ]

    ctxs: list[dict] = []
    for i in range(n_ctx):
        if i == needle_chunk:
            needle_body = (
                f"Internal only: the answer for auditors is exactly {secret}. "
                "Do not forward externally."
            )
            text = _filler_block(rng, f"H{i}", 18) + " " + needle_body + " " + _filler_block(rng, f"T{i}", 12)
        else:
            text = _filler_block(rng, f"C{i}", 24)
        ctxs.append({"title": rng.choice(titles), "text": text})

    total = sum(len(c["text"]) for c in ctxs)
    pad_piece = " " + _filler_sentence(rng, "PAD")
    safety = 0
    idx_cycle = 0
    while total < target_chars and safety < 500_000:
        i = idx_cycle % n_ctx
        idx_cycle += 1
        if i == needle_chunk:
            continue
        ctxs[i]["text"] += pad_piece
        total += len(pad_piece)
        safety += 1

    return {
        "question": q_template,
        "answers": [secret],
        "ctxs": ctxs,
        "needle_chunk": needle_chunk,
        "nith_depth_bin": depth_bin,
        "nith_context_bin": context_bin,
        "nith_target_chars": target_chars,
        "nith_target_depth_pct": round(target_depth_pct, 2),
    }


def _make_example_random(rng: random.Random) -> dict:
    n_ctx = rng.randint(3, 6)
    needle_chunk = rng.randrange(n_ctx)
    secret, q_template = _pick_qa(rng)
    titles = [
        "Ops bulletin",
        "Legal scratch pad",
        "Runbook fragment",
        "Vendor chatter",
        "HR reminder",
        "Facilities log",
        "Engineering note",
        "Random standup",
    ]
    ctxs = []
    for i in range(n_ctx):
        if i == needle_chunk:
            needle_body = (
                f"Internal only: the answer for auditors is exactly {secret}. "
                "Do not forward externally."
            )
            text = _filler_block(rng, f"H{i}", 16) + " " + needle_body + " " + _filler_block(rng, f"T{i}", 10)
        else:
            text = _filler_block(rng, f"C{i}", 20)
        ctxs.append({"title": rng.choice(titles), "text": text})
    return {
        "question": q_template,
        "answers": [secret],
        "ctxs": ctxs,
        "needle_chunk": needle_chunk,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true", help="Legacy random sampling (sparse heatmap).")
    p.add_argument("-n", "--num", type=int, default=200, help="With --random: number of examples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth-steps", type=int, default=10, help="Grid: needle depth bins (0-100%%).")
    p.add_argument("--context-steps", type=int, default=10, help="Grid: context length bins.")
    p.add_argument("--replicas", type=int, default=2, help="Grid: examples per cell (statistical power).")
    p.add_argument(
        "--max-token-budget",
        type=int,
        default=8192,
        help="Match LONGCTX_MAX_CTX_LEN / max_ctx_len; scales max passage chars (~chars ≈ budget × 3.4).",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=3.4,
        help="Upper char estimate per token for sizing text (English-ish prose + markup).",
    )
    p.add_argument(
        "--min-chars-frac",
        type=float,
        default=0.06,
        help="Min context = max_chars * this fraction (wider X axis on heatmap).",
    )
    p.add_argument("--n-ctx", type=int, default=12, help="Grid: number of passage chunks (depth resolution).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: longctx_eval/inputs/needle_haystack_l.json",
    )
    args = p.parse_args()
    rng = random.Random(args.seed)
    root = _repo_root()
    out = args.out or (root / "longctx_eval" / "inputs" / "needle_haystack_l.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.random:
        rows = [_make_example_random(rng) for _ in range(args.num)]
        meta = {"mode": "random", "n": len(rows)}
    else:
        max_chars = int(args.max_token_budget * args.chars_per_token)
        min_chars = max(600, int(max_chars * args.min_chars_frac))
        if min_chars >= max_chars:
            min_chars = max(400, max_chars // 5)

        rows = []
        for db in range(args.depth_steps):
            for cb in range(args.context_steps):
                for _ in range(max(1, args.replicas)):
                    rows.append(
                        _make_grid_example(
                            rng,
                            depth_bin=db,
                            context_bin=cb,
                            depth_steps=args.depth_steps,
                            context_steps=args.context_steps,
                            min_chars=min_chars,
                            max_chars=max_chars,
                            n_ctx=max(4, args.n_ctx),
                        )
                    )
        meta = {
            "mode": "grid",
            "depth_steps": args.depth_steps,
            "context_steps": args.context_steps,
            "replicas": args.replicas,
            "max_token_budget": args.max_token_budget,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "chars_per_token_assumption": args.chars_per_token,
            "n_ctx": max(4, args.n_ctx),
            "n_examples": len(rows),
        }

    out_meta = out.with_name(out.stem + ".grid_meta.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} examples -> {out}")
    print(f"Wrote grid metadata -> {out_meta}")


if __name__ == "__main__":
    main()
