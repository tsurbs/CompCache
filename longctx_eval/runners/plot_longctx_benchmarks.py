"""Publication-style figures for long-context benchmarks (NITH depth curve, Oolong-style bars/scatters)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_rows(dataset_path: str) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def _needle_depth_percent(ex: dict) -> float:
    n = len(ex.get("ctxs") or [])
    if n <= 1:
        return 50.0
    idx = int(ex.get("needle_chunk", 0))
    idx = min(max(idx, 0), n - 1)
    return 100.0 * idx / (n - 1)


def save_nith_figures(
    dataset_path: str,
    result: dict[str, Any],
    *,
    metric_name: str = "F1",
    blend_label: str = "CacheBlend",
    full_label: str = "Full prefill",
) -> tuple[str | None, str | None]:
    """Needle-in-haystack: depth vs quality + TTFT, binned depth curve (typical NITH look)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("NITH figures skipped: install matplotlib and numpy for charts.")
        return None, None

    key_b = f"{metric_name}_blend"
    key_f = f"{metric_name}_full"
    yb = np.asarray(result[key_b], dtype=float)
    yf = np.asarray(result[key_f], dtype=float)
    ttft_b = np.asarray(result["ttft_blend"], dtype=float) * 1e3
    ttft_f = np.asarray(result["ttft_full"], dtype=float) * 1e3

    rows = _load_rows(dataset_path)
    if len(rows) != len(yb):
        rows = rows[: len(yb)]

    depths = np.asarray([_needle_depth_percent(ex) for ex in rows], dtype=float)
    base = Path(dataset_path).resolve()
    png_path = base.with_name(f"{base.stem}_nith.png")
    json_path = base.with_name(f"{base.stem}_nith.json")

    n_bins = min(8, max(3, len(depths) // 2))
    raw = np.floor(depths / 100.0 * n_bins).astype(int)
    bin_idx = np.clip(raw, 0, n_bins - 1)
    bin_centers = (np.arange(n_bins) + 0.5) * (100.0 / n_bins)
    bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
    mean_b, mean_f, counts = [], [], []
    for bi in range(n_bins):
        mask = bin_idx == bi
        counts.append(int(mask.sum()))
        if mask.any():
            mean_b.append(float(yb[mask].mean()))
            mean_f.append(float(yf[mask].mean()))
        else:
            mean_b.append(float("nan"))
            mean_f.append(float("nan"))

    payload = {
        "benchmark": "needle_in_haystack",
        "dataset": str(base),
        "n_examples": len(yb),
        "depth_bins_percent": {
            "edges": [float(x) for x in bin_edges],
            "centers": [float(x) for x in bin_centers],
            "count_per_bin": counts,
            f"mean_{metric_name}_blend": mean_b,
            f"mean_{metric_name}_full": mean_f,
        },
        "per_example": [
            {
                "depth_percent": float(depths[i]),
                "needle_chunk": rows[i].get("needle_chunk"),
                f"{metric_name}_blend": float(yb[i]),
                f"{metric_name}_full": float(yf[i]),
                "ttft_ms_blend": float(ttft_b[i]),
                "ttft_ms_full": float(ttft_f[i]),
            }
            for i in range(len(yb))
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    color_blend = "#2563eb"
    color_full = "#dc2626"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.suptitle("Needle in a haystack — depth vs performance (chunk order proxy)", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    mean_b_arr = np.asarray(mean_b, dtype=float)
    mean_f_arr = np.asarray(mean_f, dtype=float)
    valid = ~np.isnan(mean_b_arr) & ~np.isnan(mean_f_arr)
    ax.plot(
        bin_centers[valid],
        mean_b_arr[valid],
        "o-",
        color=color_blend,
        linewidth=2,
        markersize=7,
        label=blend_label,
    )
    ax.plot(
        bin_centers[valid],
        mean_f_arr[valid],
        "s-",
        color=color_full,
        linewidth=2,
        markersize=6,
        label=full_label,
    )
    ax.set_xlabel("Needle position (% through passage chunks, document order)")
    ax.set_ylabel(f"Mean {metric_name} (binned)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title("Binned curve (typical NITH-style depth sweep)")

    ax = axes[0, 1]
    jitter = 1.2
    ax.scatter(depths + jitter * 0, yb, alpha=0.75, s=56, c=color_blend, edgecolors="white", linewidths=0.6, label=blend_label)
    ax.scatter(depths + jitter, yf, alpha=0.75, s=56, c=color_full, marker="s", edgecolors="white", linewidths=0.6, label=full_label)
    ax.set_xlabel("Needle position (% through chunks)")
    ax.set_ylabel(f"{metric_name} (per example)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax.set_title("Per-example scores (horizontal jitter for visibility)")

    ax = axes[1, 0]
    ax.scatter(depths, ttft_b, alpha=0.75, s=46, c=color_blend, label=f"{blend_label} TTFT")
    ax.scatter(depths, ttft_f, alpha=0.75, s=46, c=color_full, marker="x", label=f"{full_label} TTFT")
    ax.set_xlabel("Needle position (% through chunks)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.set_title("Latency vs needle depth")

    ax = axes[1, 1]
    xpos = np.arange(2)
    means = [float(yb.mean()), float(yf.mean())]
    stds = [float(yb.std(ddof=1)) if len(yb) > 1 else 0.0, float(yf.std(ddof=1)) if len(yf) > 1 else 0.0]
    ax.bar(xpos, means, yerr=stds, capsize=5, color=[color_blend, color_full], alpha=0.85, width=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels([blend_label, full_label], rotation=15, ha="right")
    ax.set_ylabel(f"Mean {metric_name} ± std")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_title("Overall aggregate")

    fig.savefig(png_path, dpi=175, facecolor="white")
    plt.close(fig)
    print(f"NITH benchmark figure: {png_path}")
    print(f"NITH benchmark data: {json_path}")
    return str(png_path), str(json_path)


def save_nith_heatmaps(
    dataset_path: str,
    result: dict[str, Any],
    *,
    metric_name: str = "F1",
    blend_label: str = "CacheBlend",
    full_label: str = "Full prefill",
    depth_bins: int = 10,
    context_bins: int = 10,
    success_threshold: float = 0.5,
) -> tuple[str | None, str | None]:
    """Classic green/red NITH grids: needle depth × context length, colored by mean score.

    Matches the usual benchmark presentation (depth vs context extent, pass/fail or
    continuous quality as hue). Uses ``RdYlGn`` so low scores read as red and high as green.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy.ma as ma
        from matplotlib.colors import Normalize
    except ImportError:
        print("NITH heatmaps skipped: install matplotlib and numpy for charts.")
        return None, None

    key_b = f"{metric_name}_blend"
    key_f = f"{metric_name}_full"
    yb = np.asarray(result[key_b], dtype=float)
    yf = np.asarray(result[key_f], dtype=float)

    rows = _load_rows(dataset_path)
    if len(rows) != len(yb):
        rows = rows[: len(yb)]

    depths = np.asarray([_needle_depth_percent(ex) for ex in rows], dtype=float)
    ctx = np.asarray([_context_chars(ex) for ex in rows], dtype=float)

    base = Path(dataset_path).resolve()
    png_path = base.with_name(f"{base.stem}_nith_heatmap.png")
    json_path = base.with_name(f"{base.stem}_nith_heatmap.json")
    grid_meta_path = base.with_name(f"{base.stem}.grid_meta.json")
    grid_meta: dict[str, Any] = {}
    if grid_meta_path.is_file():
        with open(grid_meta_path, encoding="utf-8") as f:
            grid_meta = json.load(f)

    use_explicit_bins = (
        grid_meta.get("mode") == "grid"
        and len(rows) > 0
        and all(
            isinstance(r.get("nith_depth_bin"), int) and isinstance(r.get("nith_context_bin"), int)
            for r in rows[: min(len(rows), len(yb))]
        )
    )

    if use_explicit_bins:
        ds = int(grid_meta["depth_steps"])
        cs = int(grid_meta["context_steps"])
        depth_edges = np.linspace(0.0, 100.0, ds + 1)
        ctx_edges = np.linspace(float(grid_meta["min_chars"]), float(grid_meta["max_chars"]), cs + 1)
        ny, nx = ds, cs

        def _build_grid(yscore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s = np.zeros((ny, nx), dtype=float)
            cnt = np.zeros((ny, nx), dtype=int)
            for i in range(min(len(rows), len(yscore))):
                db = int(rows[i]["nith_depth_bin"])
                cb = int(rows[i]["nith_context_bin"])
                db = min(max(db, 0), ny - 1)
                cb = min(max(cb, 0), nx - 1)
                s[db, cb] += float(yscore[i])
                cnt[db, cb] += 1
            g = np.full((ny, nx), np.nan, dtype=float)
            mask = cnt > 0
            g[mask] = s[mask] / cnt[mask]
            return g, cnt
    else:
        depth_edges = np.linspace(0.0, 100.0, depth_bins + 1)
        cmin, cmax = float(np.min(ctx)), float(np.max(ctx))
        if not np.isfinite(cmin):
            cmin = 0.0
        if cmax <= cmin:
            cmax = cmin + 1.0
        ctx_edges = np.linspace(cmin, cmax, context_bins + 1)
        ny, nx = depth_bins, context_bins

        def _assign_depth_bin(d: float) -> int:
            j = int(np.floor((d / 100.0) * depth_bins))
            return min(max(j, 0), depth_bins - 1)

        def _assign_ctx_bin(c: float) -> int:
            j = int(np.searchsorted(ctx_edges, c, side="right") - 1)
            return min(max(j, 0), context_bins - 1)

        def _build_grid(yscore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s = np.zeros((ny, nx), dtype=float)
            cnt = np.zeros((ny, nx), dtype=int)
            for i in range(min(len(rows), len(yscore))):
                di = _assign_depth_bin(float(depths[i]))
                ci = _assign_ctx_bin(float(ctx[i]))
                s[di, ci] += float(yscore[i])
                cnt[di, ci] += 1
            g = np.full((ny, nx), np.nan, dtype=float)
            mask = cnt > 0
            g[mask] = s[mask] / cnt[mask]
            return g, cnt

    grid_b, cnt_b = _build_grid(yb)
    grid_f, cnt_f = _build_grid(yf)
    hit_b, _ = _build_grid((yb >= success_threshold).astype(float))
    hit_f, _ = _build_grid((yf >= success_threshold).astype(float))

    payload = {
        "benchmark": "needle_in_haystack_heatmap",
        "dataset": str(base),
        "n_examples": len(yb),
        "success_threshold": success_threshold,
        "depth_bin_edges_percent": [float(x) for x in depth_edges],
        "context_bin_edges_chars": [float(x) for x in ctx_edges],
        f"grid_mean_{metric_name}_blend": [[float(v) if not np.isnan(v) else None for v in row] for row in grid_b],
        f"grid_mean_{metric_name}_full": [[float(v) if not np.isnan(v) else None for v in row] for row in grid_f],
        "grid_success_rate_blend": [[float(v) if not np.isnan(v) else None for v in row] for row in hit_b],
        "grid_success_rate_full": [[float(v) if not np.isnan(v) else None for v in row] for row in hit_f],
        "counts_blend": cnt_b.tolist(),
        "counts_full": cnt_f.tolist(),
        "used_explicit_grid_bins": use_explicit_bins,
        "grid_meta": grid_meta if grid_meta else None,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), layout="constrained")
    fig.suptitle(
        "Needle in a haystack — depth × context (green = stronger, red = weaker)",
        fontsize=14,
        fontweight="bold",
    )

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#e8e8e8")

    for ax, grid, title in (
        (axes[0, 0], grid_b, f"{blend_label} — mean {metric_name}"),
        (axes[0, 1], grid_f, f"{full_label} — mean {metric_name}"),
        (axes[1, 0], hit_b, f"{blend_label} — success rate (≥{success_threshold:g})"),
        (axes[1, 1], hit_f, f"{full_label} — success rate (≥{success_threshold:g})"),
    ):
        im = ax.imshow(
            ma.masked_invalid(grid),
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[
                float(ctx_edges[0]),
                float(ctx_edges[-1]),
                float(depth_edges[0]),
                float(depth_edges[-1]),
            ],
        )
        ax.set_xlabel("Context size (characters in passages)")
        ax.set_ylabel("Needle depth (% along passage order)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.grid(False)

    if grid_meta.get("max_token_budget"):
        fig.supxlabel(
            f"Context axis sized for ~{grid_meta['max_token_budget']} token trim budget "
            f"({grid_meta.get('min_chars')}–{grid_meta.get('max_chars')} target chars; "
            f"grid {grid_meta.get('depth_steps')}×{grid_meta.get('context_steps')})",
            fontsize=10,
        )

    fig.savefig(png_path, dpi=175, facecolor="white")
    plt.close(fig)
    print(f"NITH heatmap figure: {png_path}")
    print(f"NITH heatmap data: {json_path}")
    return str(png_path), str(json_path)


def _context_chars(ex: dict) -> int:
    return sum(len((c.get("text") or "")) for c in (ex.get("ctxs") or []))


def save_oolong_style_figures(
    dataset_path: str,
    result: dict[str, Any],
    *,
    metric_name: str = "F1",
    blend_label: str = "CacheBlend",
    full_label: str = "Full prefill",
) -> tuple[str | None, str | None]:
    """Oolong-inspired: long-context aggregation — context size vs quality + aggregate bars."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Oolong-style figures skipped: install matplotlib and numpy for charts.")
        return None, None

    key_b = f"{metric_name}_blend"
    key_f = f"{metric_name}_full"
    yb = np.asarray(result[key_b], dtype=float)
    yf = np.asarray(result[key_f], dtype=float)

    rows = _load_rows(dataset_path)
    if len(rows) != len(yb):
        rows = rows[: len(yb)]

    chars = np.asarray([_context_chars(ex) for ex in rows], dtype=float)
    k_chars = chars / 1000.0

    base = Path(dataset_path).resolve()
    png_path = base.with_name(f"{base.stem}_oolong_style.png")
    json_path = base.with_name(f"{base.stem}_oolong_style.json")

    order = np.argsort(chars)
    n_ex = len(chars)
    q_mean_b, q_mean_f, q_labels = [], [], []
    for qi in range(4):
        lo_i = qi * n_ex // 4
        hi_i = (qi + 1) * n_ex // 4 if qi < 3 else n_ex
        sl = order[lo_i:hi_i]
        if sl.size == 0:
            q_mean_b.append(float("nan"))
            q_mean_f.append(float("nan"))
            q_labels.append(f"Q{qi + 1}")
            continue
        q_mean_b.append(float(yb[sl].mean()))
        q_mean_f.append(float(yf[sl].mean()))
        c_sub = chars[sl]
        q_labels.append(f"Q{qi + 1}\n[{int(c_sub.min())}-{int(c_sub.max())}]ch")

    payload = {
        "benchmark": "oolong_style",
        "dataset": str(base),
        "n_examples": len(yb),
        "overall": {
            f"mean_{metric_name}_blend": float(yb.mean()),
            f"mean_{metric_name}_full": float(yf.mean()),
            f"std_{metric_name}_blend": float(yb.std(ddof=1)) if len(yb) > 1 else 0.0,
            f"std_{metric_name}_full": float(yf.std(ddof=1)) if len(yf) > 1 else 0.0,
        },
        "by_context_quartile": [
            {"label": q_labels[i], f"mean_{metric_name}_blend": q_mean_b[i], f"mean_{metric_name}_full": q_mean_f[i]}
            for i in range(4)
        ],
        "per_example": [
            {
                "context_chars": int(chars[i]),
                f"{metric_name}_blend": float(yb[i]),
                f"{metric_name}_full": float(yf[i]),
            }
            for i in range(len(yb))
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    color_blend = "#2563eb"
    color_full = "#dc2626"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.suptitle(
        "Long-context aggregation (Oolong-style) — context vs quality",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    xpos = np.arange(4)
    w = 0.36
    qb = np.asarray(q_mean_b, dtype=float)
    qf = np.asarray(q_mean_f, dtype=float)
    ax.bar(xpos - w / 2, qb, width=w, label=blend_label, color=color_blend, alpha=0.88)
    ax.bar(xpos + w / 2, qf, width=w, label=full_label, color=color_full, alpha=0.88)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"Q{i+1}" for i in range(4)])
    ax.set_ylabel(f"Mean {metric_name}")
    ax.set_xlabel("Context length quartile (character count)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.set_title("Performance by context size quartile")

    ax = axes[0, 1]
    ax.scatter(k_chars, yb, alpha=0.72, s=50, c=color_blend, label=blend_label, edgecolors="white", linewidths=0.5)
    ax.scatter(k_chars, yf, alpha=0.72, s=46, c=color_full, marker="s", label=full_label, edgecolors="white", linewidths=0.5)
    for i in range(len(k_chars)):
        ax.plot([k_chars[i], k_chars[i]], [yb[i], yf[i]], color="#94a3b8", linewidth=0.8, alpha=0.55, zorder=0)
    ax.set_xlabel("Total context (thousands of characters)")
    ax.set_ylabel(f"{metric_name}")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Per-example: context length vs score (paired lines)")

    ax = axes[1, 0]
    idx = np.arange(len(yb))
    ax.plot(idx, yb[order], "o-", color=color_blend, label=blend_label, alpha=0.85)
    ax.plot(idx, yf[order], "s-", color=color_full, label=full_label, alpha=0.85)
    ax.set_xlabel("Examples sorted by context length →")
    ax.set_ylabel(f"{metric_name}")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Sorted strip: difficulty / length trend")

    ax = axes[1, 1]
    xpos = np.arange(2)
    means = [float(yb.mean()), float(yf.mean())]
    stds = [float(yb.std(ddof=1)) if len(yb) > 1 else 0.0, float(yf.std(ddof=1)) if len(yf) > 1 else 0.0]
    ax.bar(xpos, means, yerr=stds, capsize=5, color=[color_blend, color_full], alpha=0.85, width=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([blend_label, full_label], rotation=12, ha="right")
    ax.set_ylabel(f"Mean {metric_name} ± std")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_title("Overall mean (aggregation benchmark summary)")

    fig.savefig(png_path, dpi=175, facecolor="white")
    plt.close(fig)
    print(f"Oolong-style figure: {png_path}")
    print(f"Oolong-style data: {json_path}")
    return str(png_path), str(json_path)
