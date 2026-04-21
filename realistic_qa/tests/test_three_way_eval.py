"""Contract tests for the three-way evaluator (Full / CompCache-single / CompCache+pairs).

Source-level checks only — actual eval requires vLLM + GPU and runs on Modal
(see ``modal run modal_runner.py::standard_3way`` and ``::realistic --mode 3way``).
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RUNNERS = _REPO / "realistic_qa" / "runners"
_THREE_WAY = _RUNNERS / "three_way_eval.py"
_BLEND = _RUNNERS / "blend_realistic.py"
_STD_UTILS = _REPO / "standard_qa" / "runners" / "utils.py"


def test_three_way_module_defines_run_blend_eval_three_way():
    s = _THREE_WAY.read_text(encoding="utf-8")
    assert "def run_blend_eval_three_way(" in s
    assert "disable_pairs=True" in s, "Method 2 must call assemble(disable_pairs=True)"
    assert "_3way_scores.json" in s
    assert "_3way_coretrieval.json" in s
    assert "_3way_ttft_warmup" in s
    assert "_3way_ttft_hist" in s


def test_three_way_payload_keys_documented_in_source():
    """Per-method keys must remain stable (downstream tables / plots key off these)."""
    s = _THREE_WAY.read_text(encoding="utf-8")
    for key in (
        '"ttft_full"',
        '"ttft_single"',
        '"ttft_pair"',
        '"collect_seconds_full"',
        '"collect_seconds_single"',
        '"collect_seconds_pair"',
        '"total_seconds_full"',
        '"total_seconds_single"',
        '"total_seconds_pair"',
        '"per_query_single_stats"',
        '"per_query_pair_stats"',
        '"single_fifo_kv"',
        '"pair_fifo_kv"',
        '"pair_store"',
        '"pair_mode"',
    ):
        assert key in s, f"missing payload key in three_way_eval: {key}"


def test_three_way_records_three_independent_methods():
    s = _THREE_WAY.read_text(encoding="utf-8")
    assert "single_fifo = FIFOChunkKVCache" in s, "Method 2 needs its own FIFO"
    assert "pair_fifo = FIFOChunkKVCache" in s, (
        "Realistic 3-way must still create a pair FIFO for the +pairs method"
    )
    assert "_run_full_prefill" in s, "Method 1 must invoke a no-cache prefill helper"
    assert "_run_cached_prefill" in s, "Methods 2 and 3 must invoke a cached prefill helper"


def test_three_way_realistic_pairs_uses_treat_all_cached_fifo_store():
    s = _THREE_WAY.read_text(encoding="utf-8")
    assert "treat_all_pairs_as_cached=True" in s, (
        "realistic +pairs path must bypass the CoRetrievalLogger"
    )
    assert "fifo=True" in s, (
        "realistic +pairs pair store must be constructed as a FIFO"
    )


def test_three_way_standard_pairs_uses_per_query_no_cache_assembler():
    s = _THREE_WAY.read_text(encoding="utf-8")
    assert "from per_query_pair_assembler import assemble_pairs_per_query" in s
    assert "assemble_pairs_per_query(" in s, (
        "standard_qa 3-way +pairs must call assemble_pairs_per_query"
    )
    # Dispatch must branch on standard_qa.
    assert "if standard_qa:" in s


def test_blend_realistic_dispatches_three_way_mode():
    s = _BLEND.read_text(encoding="utf-8")
    assert "from three_way_eval import run_blend_eval_three_way" in s
    assert 'mode == "3way"' in s
    assert "run_blend_eval_three_way(" in s


def test_standard_utils_exposes_run_blend_eval_three_way():
    s = _STD_UTILS.read_text(encoding="utf-8")
    assert "def run_blend_eval_three_way(" in s
    assert "from three_way_eval import run_blend_eval_three_way" in s
    assert '"standard_3way"' in s or 'standard_qa=True' in s


def test_three_way_eval_uses_separate_collect_timing_per_method():
    """Method 1 records collect=0; Methods 2 and 3 record perf_counter deltas."""
    s = _THREE_WAY.read_text(encoding="utf-8")
    assert "collect_full.append(0.0)" in s, (
        "Full prefill should record zero chunk-collect time"
    )
    assert "time.perf_counter()" in s, "Methods 2 and 3 should time the assemble phase"
