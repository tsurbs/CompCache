"""Side-by-side comparison plot: Full-Joint vs Sparse-Delta pair store.

Reads two ``_scores.json`` artifacts produced by
``standard_qa/runners/blend_<dataset>_comp.py`` (Full-Joint baseline) and
``standard_qa/runners/blend_<dataset>_comp_delta.py`` (Sparse-Delta), and
emits a three-panel PNG summarising:

1. Mean quality metric (F1 / ROUGE-L): Full-Prefill vs Full-Joint vs
   Sparse-Delta.  The Full-Prefill column is identical to Full-Joint's
   ``mean_<metric>_full`` (both runs saw the same dataset order under
   ``shuffle_dataset=False``) so we show it once as the upper bound.
2. Pair-store memory footprint in MB (``pair_store.bytes_used`` at end of
   run) for the two cached variants — this is the delta-caching savings.
3. Per-query TTFT distribution (violin) for Full-Prefill vs Full-Joint
   vs Sparse-Delta so latency regressions from reconstruction are
   visible.

Usage
-----
.. code:: bash

    python plot_delta_vs_full.py \
        --full  standard_qa/inputs/multihop_rag_s_comp_scores.json \
        --delta standard_qa/inputs/multihop_rag_s_comp_delta_scores.json \
        --out   standard_qa/inputs/multihop_rag_s_comp_delta_vs_full.png

If ``--out`` is omitted it defaults to
``<dataset_stem>_comp_delta_vs_full.png`` next to the input files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Headless backend so this also works inside Modal containers.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _metric_name(payload: dict[str, Any]) -> str:
    name = payload.get("metric_name")
    if isinstance(name, str) and name:
        return name
    # Fallback: sniff the per-query keys.
    for k in payload:
        if k.startswith("mean_") and k.endswith("_blend"):
            return k[len("mean_"): -len("_blend")]
    return "score"


def _bytes_used_mb(payload: dict[str, Any]) -> float:
    ps = payload.get("pair_store") or {}
    return float(ps.get("bytes_used", 0)) / (1024.0 ** 2)


def _pair_store_label(payload: dict[str, Any]) -> str:
    kind = (payload.get("pair_store_kind") or "full").lower()
    if kind in ("delta", "sparse", "sparse_delta"):
        r = payload.get("delta_top_k_ratio")
        tag = f" r={float(r):.2f}" if r is not None else ""
        return f"Sparse-Delta{tag}"
    return "Full-Joint"


def _as_float_list(payload: dict[str, Any], key: str) -> list[float]:
    raw = payload.get(key) or []
    out: list[float] = []
    for v in raw:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def plot_comparison(
    full_path: str,
    delta_path: str,
    out_path: str,
    *,
    dataset_label: str | None = None,
) -> None:
    full = _load(full_path)
    delta = _load(delta_path)

    metric = _metric_name(full)
    if metric != _metric_name(delta):
        print(
            f"[plot_delta_vs_full] WARNING: metric mismatch "
            f"({metric!r} vs {_metric_name(delta)!r}); using {metric!r}",
            file=sys.stderr,
        )

    mean_full_prefill = float(
        full.get(f"mean_{metric}_full", np.nan)
    )
    mean_full_joint = float(
        full.get(f"mean_{metric}_blend", np.nan)
    )
    mean_delta = float(
        delta.get(f"mean_{metric}_blend", np.nan)
    )

    full_joint_label = _pair_store_label(full)
    delta_label = _pair_store_label(delta)

    full_bytes_mb = _bytes_used_mb(full)
    delta_bytes_mb = _bytes_used_mb(delta)
    n_queries_full = int(full.get("n_queries", 0))
    n_queries_delta = int(delta.get("n_queries", 0))

    ttft_prefill = _as_float_list(full, "ttft_full")
    ttft_full_joint = _as_float_list(full, "ttft_blend")
    ttft_delta = _as_float_list(delta, "ttft_blend")

    if dataset_label is None:
        ds_path = full.get("dataset") or full_path
        dataset_label = Path(ds_path).stem.replace("_comp_scores", "")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: mean quality ----------------------------------------
    ax = axes[0]
    labels = ["Full Prefill", full_joint_label, delta_label]
    values = [mean_full_prefill, mean_full_joint, mean_delta]
    colors = ["#4b5d67", "#1f77b4", "#d62728"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylabel(f"mean {metric.upper()}")
    ax.set_title(f"Quality — {dataset_label}")
    ax.set_ylim(0, max([v for v in values if not np.isnan(v)] + [0.1]) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Relative drop vs Full-Joint as subtitle.
    if not np.isnan(mean_full_joint) and mean_full_joint > 0 and not np.isnan(mean_delta):
        rel = (mean_delta - mean_full_joint) / mean_full_joint * 100.0
        sign = "+" if rel >= 0 else ""
        ax.text(
            2,
            mean_delta,
            f"  {sign}{rel:.1f}% vs Full-Joint",
            fontsize=8,
            color="dimgray",
            va="bottom",
        )

    # --- Panel 2: pair-store memory -----------------------------------
    ax = axes[1]
    mem_labels = [full_joint_label, delta_label]
    mem_values = [full_bytes_mb, delta_bytes_mb]
    bars = ax.bar(
        mem_labels,
        mem_values,
        color=["#1f77b4", "#d62728"],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, v in zip(bars, mem_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("pair_store.bytes_used (MB)")
    ax.set_title("Pair-store memory footprint")
    ax.grid(axis="y", alpha=0.3)
    if full_bytes_mb > 0 and delta_bytes_mb >= 0:
        ratio = full_bytes_mb / max(delta_bytes_mb, 1e-9)
        ax.text(
            0.5,
            0.95,
            f"{ratio:.1f}× compression",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="dimgray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, lw=0),
        )

    # --- Panel 3: TTFT distribution (violin) --------------------------
    ax = axes[2]
    data = [
        ttft_prefill if ttft_prefill else [0.0],
        ttft_full_joint if ttft_full_joint else [0.0],
        ttft_delta if ttft_delta else [0.0],
    ]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("TTFT (s)")
    ax.set_title(
        f"Per-query TTFT distribution"
        f" (n={min(n_queries_full, n_queries_delta)})"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Sparse-Delta vs Full-Joint pair store — {dataset_label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_delta_vs_full] wrote {out_path}")


def _infer_out_path(full_path: str, delta_path: str) -> str:
    d = Path(delta_path).resolve()
    stem = d.stem
    if stem.endswith("_comp_delta_scores"):
        base = stem[: -len("_comp_delta_scores")]
    elif stem.endswith("_scores"):
        base = stem[: -len("_scores")]
    else:
        base = stem
    return str(d.with_name(f"{base}_comp_delta_vs_full.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument("--full", required=True, help="Path to *_comp_scores.json (Full-Joint baseline)")
    ap.add_argument(
        "--delta",
        required=True,
        help="Path to *_comp_delta_scores.json (Sparse-Delta variant)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to <dataset>_comp_delta_vs_full.png alongside --delta.",
    )
    ap.add_argument(
        "--dataset-label",
        default=None,
        help="Override the dataset label shown in the plot title.",
    )
    args = ap.parse_args()

    out_path = args.out or _infer_out_path(args.full, args.delta)
    plot_comparison(
        args.full,
        args.delta,
        out_path,
        dataset_label=args.dataset_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
