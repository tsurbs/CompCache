"""Contract tests for CompCache (shared ``comp_blend_eval`` + ``blend_realistic`` dispatch).

These read runner sources as text so CI does not need ``rouge_score``, vLLM, or a GPU.
End-to-end CompCache eval is run manually (see ``realistic_qa/README.md``).
"""
from __future__ import annotations

from pathlib import Path

_RUNNERS = Path(__file__).resolve().parents[1] / "runners"
_BLEND = _RUNNERS / "blend_realistic.py"
_COMP = _RUNNERS / "comp_blend_eval.py"


def test_comp_module_defines_run_blend_eval_comp_and_ttft_reports():
    s = _COMP.read_text(encoding="utf-8")
    assert "def run_blend_eval_comp(" in s
    assert "_comp_coretrieval.json" in s
    assert "_comp_scores.json" in s
    assert 'name_suffix="_comp"' in s
    assert "CompCache (composition-aware)" in s
    assert "save_ttft_warmup_plot(" in s
    assert "save_ttft_histogram(" in s
    assert "per_query_stats" in s
    assert "pair_store" in s and "promotion_worker" in s


def test_blend_realistic_imports_comp_and_dispatches_main():
    s = _BLEND.read_text(encoding="utf-8")
    assert "from comp_blend_eval import" in s and "run_blend_eval_comp" in s
    assert '("REALISTIC_MODE", "fifo")' in s
    assert 'mode == "comp"' in s
    assert "run_blend_eval_comp(" in s
    assert "run_blend_eval_fifo(" in s


def test_comp_scores_payload_keys_documented_in_source():
    """Keep these keys stable for downstream plotting (e.g. plot_compare_json)."""
    s = _COMP.read_text(encoding="utf-8")
    for key in (
        '"ttft_blend"',
        '"ttft_full"',
        '"per_query_stats"',
        '"promotion_threshold"',
        '"stream_seed"',
    ):
        assert key in s
