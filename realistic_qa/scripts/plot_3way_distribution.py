#!/usr/bin/env python3
"""Render 3-way TTFT / total-time distributions from ``*_3way_scores.json``.

Produces a 2x2 figure:

- (top-left)     Prefill TTFT — overlaid histograms (distinguishable when method TTFTs differ)
- (top-right)    Prefill TTFT — boxplot per method
- (bottom-left)  Total time (TTFT + collect) — overlaid histograms
- (bottom-right) Total time — boxplot per method
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


def _ms_arr(xs):
    import numpy as np
    return np.asarray(xs, dtype=float) * 1e3


def _hist_panel(ax, data, key_template, title):
    bins = 60
    series = []
    labels = []
    colors = []
    for tag, label, color in _SERIES:
        ys = data.get(key_template.format(tag=tag)) or []
        if not ys:
            continue
        series.append(_ms_arr(ys))
        labels.append(label)
        colors.append(color)
    if not series:
        return
    for ys, label, color in zip(series, labels, colors):
        ax.hist(
            ys,
            bins=bins,
            density=True,
            alpha=0.5,
            label=label,
            color=color,
            edgecolor=color,
            linewidth=0.5,
        )
    ax.set_xlabel(f"{title} (ms)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title} — distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _box_panel(ax, data, key_template, title):
    series = []
    labels = []
    colors = []
    for tag, label, color in _SERIES:
        ys = data.get(key_template.format(tag=tag)) or []
        if not ys:
            continue
        series.append(_ms_arr(ys))
        labels.append(label)
        colors.append(color)
    if not series:
        return
    bp = ax.boxplot(
        series,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel(f"{title} (ms)")
    ax.set_title(f"{title} — boxplot (median ± IQR, whiskers to 1.5·IQR)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=8)


def render(scores_path: Path, out: Path) -> Path:
    data = json.loads(scores_path.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    _hist_panel(axes[0, 0], data, "ttft_{tag}",           "Prefill TTFT")
    _box_panel( axes[0, 1], data, "ttft_{tag}",           "Prefill TTFT")
    _hist_panel(axes[1, 0], data, "total_seconds_{tag}",  "Total time (TTFT + collect)")
    _box_panel( axes[1, 1], data, "total_seconds_{tag}",  "Total time (TTFT + collect)")

    n = data.get("n_queries", "?")
    ds = (data.get("dataset") or "").rsplit("/", 1)[-1]
    fig.suptitle(
        f"Three-way TTFT distribution · {ds} · n={n}",
        fontsize=12,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores_path", type=Path, help="Path to *_3way_scores.json")
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args()

    sp = args.scores_path.resolve()
    if args.out is None:
        stem = sp.name
        if stem.endswith("_3way_scores.json"):
            stem = stem[: -len("_3way_scores.json")]
        else:
            stem = sp.stem
        out = sp.parent / f"{stem}_3way_distribution.png"
    else:
        out = args.out
        if out.suffix.lower() != ".png":
            out = out.with_suffix(".png")

    written = render(sp, out, )
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
