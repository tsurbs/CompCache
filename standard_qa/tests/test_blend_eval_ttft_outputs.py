"""Sanity checks that standard ``run_blend_eval`` emits the same TTFT artifacts as realistic FIFO.

Reads ``utils.py`` as text so tests run without ``rouge_score`` / vLLM imports.
"""
from __future__ import annotations

from pathlib import Path

_UTILS = Path(__file__).resolve().parents[1] / "runners" / "utils.py"


def test_run_blend_eval_writes_ttft_warmup_and_histogram():
    src = _UTILS.read_text()
    assert "save_ttft_warmup_plot" in src
    assert "save_ttft_histogram" in src
    assert '"ttft_blend"' in src or "'ttft_blend'" in src
    assert '"ttft_full"' in src or "'ttft_full'" in src
    assert "CacheBlend (standard)" in src
