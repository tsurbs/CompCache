"""TTFT JSON/PNG artifacts (warmup series, histogram) — no rouge / QA dependencies."""
from __future__ import annotations

import json
from pathlib import Path


def save_ttft_histogram(
    dataset_path: str,
    ttft_blend: list[float],
    ttft_full: list[float],
    *,
    cached_label: str = "Cached (CacheBlend)",
    full_label: str = "Full prefill",
    extra_metadata: dict | None = None,
    name_suffix: str = "",
) -> tuple[str, str | None]:
    n = len(ttft_blend)
    if n == 0 or len(ttft_full) != n:
        return "", None

    base = Path(dataset_path).resolve()
    json_path = base.with_name(f"{base.stem}{name_suffix}_ttft_hist.json")
    png_path = base.with_name(f"{base.stem}{name_suffix}_ttft_hist.png")

    meta = dict(extra_metadata or {})
    meta["dataset"] = str(base)
    meta["n_queries"] = n
    payload = {
        "ttft_blend_seconds": [float(x) for x in ttft_blend],
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "metadata": meta,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    png_written: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        b_ms = np.asarray(ttft_blend, dtype=float) * 1e3
        f_ms = np.asarray(ttft_full, dtype=float) * 1e3
        edges = np.histogram_bin_edges(
            np.concatenate([b_ms, f_ms]), bins="auto"
        )
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.hist(
            b_ms,
            bins=edges,
            alpha=0.55,
            label=cached_label,
            color="#1f77b4",
            density=True,
        )
        ax.hist(
            f_ms,
            bins=edges,
            alpha=0.55,
            label=full_label,
            color="#ff7f0e",
            density=True,
        )
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("Density")
        ax.set_title("TTFT: cached vs full prefill")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        png_written = str(png_path)
    except ImportError:
        pass

    if png_written:
        print(f"TTFT histogram: {png_written}")
    print(f"TTFT histogram data: {json_path}")
    return str(json_path), png_written


def save_ttft_warmup_plot(
    dataset_path: str,
    ttft_blend: list[float],
    ttft_full: list[float],
    *,
    stream_seed: int,
    skip_first: int,
    fifo_stats: dict,
    roll_window: int,
    cached_label: str = "CacheBlend (FIFO KV)",
    name_suffix: str = "",
) -> tuple[str, str | None]:
    """Write per-query TTFT series (JSON) and a line plot (PNG) next to the dataset file.

    ``name_suffix`` is inserted before ``_ttft_warmup`` (e.g. ``_comp`` → ``*_comp_ttft_warmup.json``).
    """
    base = Path(dataset_path).resolve()
    json_path = base.with_name(f"{base.stem}{name_suffix}_ttft_warmup.json")
    png_path = base.with_name(f"{base.stem}{name_suffix}_ttft_warmup.png")

    n = len(ttft_blend)
    roll_cached = cached_label.split(" (", 1)[0]
    payload = {
        "query_index": list(range(n)),
        "ttft_blend_seconds": [float(x) for x in ttft_blend],
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "metadata": {
            "dataset": str(base),
            "stream_seed": stream_seed,
            "skip_first": skip_first,
            "fifo_kv": fifo_stats,
            "roll_window": roll_window,
            "cached_label": cached_label,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    png_written: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(n, dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.plot(x, np.asarray(ttft_blend, dtype=float) * 1e3, label=cached_label, lw=0.8, alpha=0.85)
        ax.plot(x, np.asarray(ttft_full, dtype=float) * 1e3, label="Full prefill", lw=0.8, alpha=0.85)

        if roll_window > 1 and n >= roll_window:
            w = roll_window
            k = np.ones(w, dtype=float) / w
            smooth_b = np.convolve(ttft_blend, k, mode="valid") * 1e3
            smooth_f = np.convolve(ttft_full, k, mode="valid") * 1e3
            xs = np.arange(w - 1, n, dtype=float)
            ax.plot(xs, smooth_b, label=f"{roll_cached} ({w}-query rolling mean)", lw=1.5)
            ax.plot(xs, smooth_f, label=f"Full prefill ({w}-query rolling mean)", lw=1.5)

        ax.set_xlabel("Query index (shuffled stream order)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("Time to first token vs cache warmup")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        png_written = str(png_path)
    except ImportError:
        pass

    if png_written:
        print(f"TTFT warmup plot: {png_written}")
    print(f"TTFT warmup data: {json_path}")
    return str(json_path), png_written
