"""Schema checks on committed **full** benchmark ``*_ttft_hist.json`` files under ``standard_qa/inputs/``.

These are real run outputs (e.g. MuSiQue n=150), not tiny synthetic fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

_INPUTS = Path(__file__).resolve().parents[1] / "inputs"


def _ttft_hist_paths():
    # Exclude the 3-way artifacts: they have a different schema
    # (ttft_full/ttft_single/ttft_pair instead of blend/full).
    return sorted(
        p for p in _INPUTS.glob("*_ttft_hist.json")
        if "_3way_" not in p.name
    )


def test_ttft_hist_files_exist():
    paths = _ttft_hist_paths()
    assert paths, f"expected *_ttft_hist.json under {_INPUTS}"


def test_each_ttft_hist_schema_and_lengths():
    for path in _ttft_hist_paths():
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ttft_blend_seconds" in data, path.name
        assert "ttft_full_seconds" in data, path.name
        assert "metadata" in data, path.name
        b = data["ttft_blend_seconds"]
        f = data["ttft_full_seconds"]
        assert isinstance(b, list) and isinstance(f, list), path.name
        assert len(b) == len(f), f"{path.name}: blend/full length mismatch"
        meta = data["metadata"]
        assert isinstance(meta, dict), path.name
        assert meta.get("n_queries") == len(b), path.name
        assert "dataset" in meta, path.name
