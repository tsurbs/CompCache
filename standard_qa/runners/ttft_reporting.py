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

    meta = dict(extra_metadata or {})
    meta["dataset"] = str(base)
    meta["n_queries"] = n
    meta["cached_label"] = cached_label
    meta["full_label"] = full_label
    payload = {
        "ttft_blend_seconds": [float(x) for x in ttft_blend],
        "ttft_full_seconds": [float(x) for x in ttft_full],
        "metadata": meta,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"TTFT histogram data: {json_path}")
    return str(json_path), None

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
    base = Path(dataset_path).resolve()
    json_path = base.with_name(f"{base.stem}{name_suffix}_ttft_warmup.json")

    n = len(ttft_blend)
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

    print(f"TTFT warmup data: {json_path}")
    return str(json_path), None
