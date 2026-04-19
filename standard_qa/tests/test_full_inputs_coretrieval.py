"""Sanity checks on committed **full** ``*_coretrieval.json`` files under ``standard_qa/inputs/``."""
from __future__ import annotations

import json
from pathlib import Path

_INPUTS = Path(__file__).resolve().parents[1] / "inputs"

_REQUIRED_KEYS = frozenset(
    {"alpha", "r_squared", "n_queries", "n_unique_docs", "n_unique_pairs", "top_pairs", "doc_frequencies"}
)


def _coretrieval_paths():
    return sorted(_INPUTS.glob("*_coretrieval.json"))


def test_coretrieval_files_exist():
    paths = _coretrieval_paths()
    assert paths, f"expected *_coretrieval.json under {_INPUTS}"


def test_each_coretrieval_has_expected_shape():
    for path in _coretrieval_paths():
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = _REQUIRED_KEYS - data.keys()
        assert not missing, f"{path.name} missing keys: {sorted(missing)}"
        assert isinstance(data["top_pairs"], list), path.name
        assert isinstance(data["doc_frequencies"], dict), path.name
        assert data["n_queries"] >= 1, path.name
