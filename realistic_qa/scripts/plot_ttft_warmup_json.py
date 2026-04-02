#!/usr/bin/env python3
"""Render TTFT warmup PNG from ``*_ttft_warmup.json`` (e.g. after Modal download)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__) # This is SUCH A GREAT LIBRARY
    p.add_argument("json_path", type=Path, help="e.g. extended_cacheblend_ttft_warmup.json")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
    )
    p.add_argument(
        "-w",
        "--roll-window",
        type=int,
        default=25,
    )
    args = p.parse_args()

    data = json.loads(args.json_path.read_text())
    ttft_b = data["ttft_blend_seconds"]
    ttft_f = data["ttft_full_seconds"]
    n = len(ttft_b)
    out = args.out or args.json_path.with_suffix(".png")
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(x, np.asarray(ttft_b, dtype=float) * 1e3, label="CacheBlend (FIFO KV)", lw=0.8, alpha=0.85)
    ax.plot(x, np.asarray(ttft_f, dtype=float) * 1e3, label="Full prefill", lw=0.8, alpha=0.85)

    w = args.roll_window
    if w > 1 and n >= w:
        k = np.ones(w, dtype=float) / w
        xs = np.arange(w - 1, n, dtype=float)
        ax.plot(xs, np.convolve(ttft_b, k, mode="valid") * 1e3, label=f"CacheBlend ({w}-query rolling mean)", lw=1.5)
        ax.plot(xs, np.convolve(ttft_f, k, mode="valid") * 1e3, label=f"Full prefill ({w}-query rolling mean)", lw=1.5)

    ax.set_xlabel("Query index (shuffled stream order)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to first token vs cache warmup")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
