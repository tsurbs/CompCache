#!/usr/bin/env python3
"""Render 3-way TTFT comparison from ``*_3way_scores.json``.

Overlays three series sharing the same shuffled query stream:

- **Full Prefill** — vanilla vLLM, no KV reuse
- **CompCache (single-chunk)** — selective recompute on individual chunks (no pair store)
- **CompCache (+pairs)** — composition-aware: pair-store + selective recompute

Two panels are produced:

- **Prefill TTFT** — pure model.generate latency
- **Total time** — prefill TTFT + KV collection time (the wall-clock cost of
  serving the query, where the pair-store benefit shows up vs. single-chunk)

Usage::

    python realistic_qa/scripts/plot_3way_compare.py \\
        realistic_qa/inputs/extended_cacheblend_3way_scores.json
    # → realistic_qa/inputs/extended_cacheblend_3way_compare.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_SERIES = (
    ("full",   "Full Prefill",            "#2ca02c"),
    ("single", "CompCache (single-chunk)", "#ff7f0e"),
    ("pair",   "CompCache (+pairs)",      "#1f77b4"),
)


def _rolling_mean(xs, w: int):
    import numpy as np
    if w <= 1 or len(xs) < w:
        return None, None
    k = np.ones(w, dtype=float) / w
    smoothed = np.convolve(np.asarray(xs, dtype=float), k, mode="valid") * 1e3
    x_idx = np.arange(w - 1, len(xs), dtype=float)
    return x_idx, smoothed


def _plot_panel(ax, data, key_template, title, roll_window, ylim_top=None):
    import numpy as np
    for tag, label, color in _SERIES:
        ys = data.get(key_template.format(tag=tag)) or []
        if not ys:
            continue
        x = np.arange(len(ys), dtype=float)
        ax.plot(
            x,
            np.asarray(ys, dtype=float) * 1e3,
            label=label,
            color=color,
            lw=0.7,
            alpha=0.45,
        )
    for tag, label, color in _SERIES:
        ys = data.get(key_template.format(tag=tag)) or []
        x_idx, smoothed = _rolling_mean(ys, roll_window)
        if x_idx is None:
            continue
        ax.plot(
            x_idx,
            smoothed,
            label=f"{label} ({roll_window}-query mean)",
            color=color,
            lw=2.0,
        )
    ax.set_xlabel("Query index (shuffled stream order)")
    ax.set_ylabel(f"{title} (ms)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    ax.grid(True, alpha=0.3)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)


def render(scores_path: Path, out: Path, roll_window: int) -> Path:
    data = json.loads(scores_path.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 9), layout="constrained", sharex=True
    )

    _plot_panel(
        ax_top,
        data,
        "ttft_{tag}",
        "Prefill TTFT",
        roll_window,
    )
    _plot_panel(
        ax_bot,
        data,
        "total_seconds_{tag}",
        "Total time (prefill TTFT + KV collection)",
        roll_window,
    )

    n = data.get("n_queries", "?")
    ds = (data.get("dataset") or "").rsplit("/", 1)[-1]
    fig.suptitle(
        f"Three-way TTFT comparison · {ds} · n={n}",
        fontsize=12,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "scores_path",
        type=Path,
        help="Path to *_3way_scores.json",
    )
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("-w", "--roll-window", type=int, default=25)
    args = p.parse_args()

    sp = args.scores_path.resolve()
    if args.out is None:
        stem = sp.name
        if stem.endswith("_3way_scores.json"):
            stem = stem[: -len("_3way_scores.json")]
        else:
            stem = sp.stem
        out = sp.parent / f"{stem}_3way_compare.png"
    else:
        out = args.out
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")

    written = render(sp, out, args.roll_window)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
