"""Three focused plots from one ``*_delta_memory_scores.json`` run:

1.  ``*_cache_docs.png`` — number of pairs held in each store over the
    query stream.  Includes a flat ``CacheBlend (no pair cache) = 0``
    baseline for contrast.  This is the "memory savings → more docs
    cached under the same byte budget" chart.
2.  ``*_accuracy.png`` — mean F1 per method (bars) with rolling
    per-query F1 curves overlaid so the reader can see quality stays
    flat as the cache fills.
3.  ``*_ttft_quicklook.png`` — TTFT distribution per method (violin +
    mean marker).

Usage:
    python realistic_qa/runners/plot_delta_memory_quicklook.py \
        realistic_qa/inputs/extended_cacheblend_delta_memory_scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_PALETTE = {
    "Full (joint)":     "#1f77b4",
    "Δ-sparse r=0.5":   "#ff7f0e",
    "Δ-sparse r=0.1":   "#9467bd",
    "CacheBlend (no pair cache)": "#7f7f7f",
}


def _color(name: str) -> str:
    return _PALETTE.get(name, "#2ca02c")


def _rolling_mean(values: List[float], window: int) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or window <= 1:
        return a
    kernel = np.ones(window) / window
    # ``valid`` mode avoids edge artifacts; we pad with NaN at the start
    # so the x-axis alignment is unambiguous on the plot.
    smooth = np.convolve(a, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smooth])


def plot_cache_docs(d: dict, out_png: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), layout="constrained")
    n_queries = d["n_queries"]

    for cfg in d["per_config"]:
        name = cfg["name"]
        ml = d["per_query"][name]["memory_log"]
        idx = np.array([e[0] for e in ml], dtype=float)
        entries = np.array([e[2] for e in ml], dtype=float)
        ax.plot(
            idx,
            entries,
            label=f"{name}  (final: {int(entries[-1])})",
            color=_color(name),
            lw=2.0,
        )

    # CacheBlend baseline: no pair cache → always 0.
    ax.plot(
        [0, n_queries - 1],
        [0, 0],
        label="CacheBlend (no pair cache)",
        color=_color("CacheBlend (no pair cache)"),
        ls="--",
        lw=2.0,
    )

    budget = d.get("pair_store_bytes_budget")
    budget_str = f"{budget / (1024 ** 3):.1f} GiB" if budget else "unbounded"
    ax.set_xlabel("Query index (shuffled stream order)")
    ax.set_ylabel("Documents held in pair cache (entries)")
    ax.set_title(
        f"Pair cache occupancy under {budget_str} byte budget — {dataset_label}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=10, framealpha=0.9)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_accuracy(d: dict, out_png: Path, dataset_label: str) -> None:
    metric = d.get("metric_name", "F1")
    fig, (ax_bar, ax_roll) = plt.subplots(
        1, 2, figsize=(13, 5.0), layout="constrained",
        gridspec_kw={"width_ratios": [1, 2]},
    )

    # ---- bars ----
    names = [c["name"] for c in d["per_config"]]
    means = [c[f"mean_{metric}"] for c in d["per_config"]]
    colors = [_color(n) for n in names]
    bars = ax_bar.bar(names, means, color=colors, edgecolor="black")
    for bar, m in zip(bars, means):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_bar.set_ylabel(f"Mean {metric}")
    ax_bar.set_title(f"Mean {metric} per method")
    ax_bar.set_ylim(0, max(means) * 1.18)
    ax_bar.grid(True, axis="y", alpha=0.3)
    for lbl in ax_bar.get_xticklabels():
        lbl.set_rotation(12)
        lbl.set_ha("right")

    # ---- rolling F1 curve ----
    window = max(20, d["n_queries"] // 60)
    for name in names:
        fs = d["per_query"][name][metric]
        roll = _rolling_mean(fs, window)
        ax_roll.plot(
            np.arange(len(roll)),
            roll,
            label=f"{name}",
            color=_color(name),
            lw=1.6,
            alpha=0.9,
        )
    ax_roll.set_xlabel("Query index")
    ax_roll.set_ylabel(f"Rolling mean {metric} (window={window})")
    ax_roll.set_title(
        f"Per-query {metric} over time (rolling)  — {dataset_label}"
    )
    ax_roll.grid(True, alpha=0.3)
    ax_roll.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ttft(d: dict, out_png: Path, dataset_label: str) -> None:
    names = [c["name"] for c in d["per_config"]]
    series_ms = [
        np.asarray(d["per_query"][n]["ttft"], dtype=float) * 1e3
        for n in names
    ]
    colors = [_color(n) for n in names]

    fig, (ax_v, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5.0), layout="constrained",
        gridspec_kw={"width_ratios": [2, 1]},
    )

    parts = ax_v.violinplot(
        series_ms, positions=range(len(names)), showmeans=False,
        showmedians=True, widths=0.85,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_alpha(0.55)
    for key in ("cmins", "cmaxes", "cbars", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    # Overlay mean markers.
    means_ms = [float(s.mean()) for s in series_ms]
    ax_v.scatter(
        range(len(names)), means_ms, marker="D", s=40,
        color="white", edgecolor="black", zorder=3, label="mean",
    )
    for i, m in enumerate(means_ms):
        ax_v.text(i, m, f" {m:.1f} ms", va="center", fontsize=9)
    ax_v.set_xticks(range(len(names)))
    ax_v.set_xticklabels(names, rotation=12, ha="right")
    ax_v.set_ylabel("Prefill TTFT (ms)")
    ax_v.set_title(f"TTFT distribution per method — {dataset_label}")
    ax_v.grid(True, axis="y", alpha=0.3)
    ax_v.legend(loc="upper right", fontsize=9)

    # Right panel: mean TTFT bar chart for a clean one-number read.
    bars = ax_bar.bar(names, means_ms, color=colors, edgecolor="black")
    for bar, m in zip(bars, means_ms):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{m:.1f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax_bar.set_ylabel("Mean TTFT (ms)")
    ax_bar.set_title("Mean TTFT")
    ax_bar.grid(True, axis="y", alpha=0.3)
    for lbl in ax_bar.get_xticklabels():
        lbl.set_rotation(12)
        lbl.set_ha("right")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_json")
    args = ap.parse_args()

    src = Path(args.scores_json).resolve()
    d = json.loads(src.read_text())
    stem = src.name.replace("_delta_memory_scores.json", "")
    dataset_label = stem
    out_dir = src.parent

    out1 = out_dir / f"{stem}_delta_memory_cache_docs.png"
    out2 = out_dir / f"{stem}_delta_memory_accuracy.png"
    out3 = out_dir / f"{stem}_delta_memory_ttft_quicklook.png"

    plot_cache_docs(d, out1, dataset_label)
    plot_accuracy(d, out2, dataset_label)
    plot_ttft(d, out3, dataset_label)
    print(f"wrote {out1}")
    print(f"wrote {out2}")
    print(f"wrote {out3}")


if __name__ == "__main__":
    main()
