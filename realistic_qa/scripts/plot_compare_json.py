#!/usr/bin/env python3
"""Render a head-to-head TTFT plot for ``modal_runner.py::realistic_compare`` outputs.

Overlays three series sharing the same shuffled query stream:

- **CacheBlend (FIFO KV)** — from ``<stem>_ttft_warmup.json`` (FIFO leg)
- **CompCache (composition-aware)** — from ``<stem>_comp_scores.json`` (comp leg)
- **Full prefill** — taken from the FIFO leg's ``ttft_full_seconds`` (with the comp
  leg's ``ttft_full`` shown as a dashed control if it differs in length)

Usage::

    python realistic_qa/scripts/plot_compare_json.py \\
        realistic_qa/inputs/extended_cacheblend.json
    # → realistic_qa/inputs/extended_cacheblend_compare.png

You may pass any of: the dataset path, the ``_compare.json``, the
``_ttft_warmup.json``, the ``_comp_scores.json``, or the stem itself; the script
locates siblings by stem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple


_KNOWN_SUFFIXES = (
    "_compare.json",
    "_comp_ttft_warmup.json",
    "_ttft_warmup.json",
    "_comp_scores.json",
    "_scores.json",
    ".json",
)


def _resolve_stem(path: Path) -> Tuple[Path, str]:
    """Return (parent_dir, stem) by stripping any known suffix."""
    name = path.name
    for sfx in _KNOWN_SUFFIXES:
        if name.endswith(sfx):
            return path.parent, name[: -len(sfx)]
    return path.parent, path.stem


def _load_series(parent: Path, stem: str) -> dict:
    fifo_warmup = parent / f"{stem}_ttft_warmup.json"
    comp_scores = parent / f"{stem}_comp_scores.json"
    out: dict = {}
    if fifo_warmup.is_file():
        data = json.loads(fifo_warmup.read_text())
        out["fifo_blend"] = list(data.get("ttft_blend_seconds", []) or [])
        out["full_prefill_fifo"] = list(data.get("ttft_full_seconds", []) or [])
    else:
        print(f"warning: no {fifo_warmup.name}; FIFO + full-prefill series will be missing", file=sys.stderr)
    if comp_scores.is_file():
        data = json.loads(comp_scores.read_text())
        out["comp_blend"] = list(data.get("ttft_blend", []) or [])
        out["full_prefill_comp"] = list(data.get("ttft_full", []) or [])
    else:
        print(f"warning: no {comp_scores.name}; CompCache series will be missing", file=sys.stderr)
    return out


def _rolling_mean(xs, w: int):
    import numpy as np
    if w <= 1 or len(xs) < w:
        return None, None
    k = np.ones(w, dtype=float) / w
    smoothed = np.convolve(np.asarray(xs, dtype=float), k, mode="valid") * 1e3
    x_idx = np.arange(w - 1, len(xs), dtype=float)
    return x_idx, smoothed


def render(parent: Path, stem: str, out: Path, roll_window: int) -> Path:
    series = _load_series(parent, stem)
    if not any(series.values()):
        raise FileNotFoundError(f"No TTFT series found next to '{parent / stem}*'")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")

    # Raw lines (thin, faded) — CompCache, FIFO blend, full prefill.
    color_map = {
        "comp_blend": ("CompCache (composition-aware)", "#1f77b4"),
        "fifo_blend": ("CacheBlend (FIFO KV)", "#ff7f0e"),
        "full_prefill_fifo": ("Full prefill", "#2ca02c"),
    }
    for key, (label, color) in color_map.items():
        ys = series.get(key) or []
        if not ys:
            continue
        x = np.arange(len(ys), dtype=float)
        ax.plot(x, np.asarray(ys, dtype=float) * 1e3, label=label, color=color, lw=0.7, alpha=0.5)

    # Rolling means (bold) — same colors.
    for key, (label, color) in color_map.items():
        ys = series.get(key) or []
        x_idx, smoothed = _rolling_mean(ys, roll_window)
        if x_idx is None:
            continue
        ax.plot(x_idx, smoothed, label=f"{label} ({roll_window}-query mean)", color=color, lw=2.0)

    # Comp-leg full prefill as dashed sanity check (if both exist and differ in length, useful).
    cf = series.get("full_prefill_comp") or []
    ff = series.get("full_prefill_fifo") or []
    if cf and ff and len(cf) != len(ff):
        x = np.arange(len(cf), dtype=float)
        ax.plot(x, np.asarray(cf, dtype=float) * 1e3, label="Full prefill (comp run)", color="#2ca02c", lw=0.7, ls="--", alpha=0.5)

    ax.set_xlabel("Query index (shuffled stream order)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT — CompCache vs CacheBlend(FIFO) vs Full prefill")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        type=Path,
        help="Dataset path, _compare.json, _ttft_warmup.json, or _comp_scores.json (any sibling).",
    )
    p.add_argument("-o", "--out", type=Path, default=None, help="Output PNG (default: <stem>_compare.png).")
    p.add_argument("-w", "--roll-window", type=int, default=25, help="Rolling-mean window in queries (default 25).")
    args = p.parse_args()

    parent, stem = _resolve_stem(args.path.resolve())
    out = args.out or (parent / f"{stem}_compare.png")
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    written = render(parent, stem, out, args.roll_window)
    print(f"Wrote {written.resolve()}")


if __name__ == "__main__":
    main()
