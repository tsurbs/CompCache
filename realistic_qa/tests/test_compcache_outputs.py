"""TTFT JSON/PNG helpers used by realistic CompCache and FIFO (``ttft_reporting``).

For contracts on ``run_blend_eval_comp`` itself, see ``test_blend_realistic_compcache.py``.
Uses small synthetic TTFT series and temp files — no GPU or vLLM.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_STANDARD_RUNNERS = _REPO / "standard_qa" / "runners"
sys.path.insert(0, str(_STANDARD_RUNNERS))

from ttft_reporting import save_ttft_histogram, save_ttft_warmup_plot  # noqa: E402


def _tiny_ttft_series(n: int = 5):
    base = 0.04
    blend = [base + i * 1e-4 for i in range(n)]
    full = [base + 0.01 + i * 1e-4 for i in range(n)]
    return blend, full


def test_save_ttft_histogram_name_suffix_writes_json(tmp_path: Path):
    ds = tmp_path / "mini.json"
    ds.write_text("[]")
    blend, full = _tiny_ttft_series()
    j_path, png_path = save_ttft_histogram(
        str(ds),
        blend,
        full,
        cached_label="CompCache (composition-aware)",
        name_suffix="_comp",
        extra_metadata={"stream_seed": 1},
    )
    assert j_path.endswith("mini_comp_ttft_hist.json")
    hist = json.loads(Path(j_path).read_text())
    assert hist["ttft_blend_seconds"] == blend
    assert hist["ttft_full_seconds"] == full
    assert hist["metadata"]["stream_seed"] == 1
    assert png_path and Path(png_path).is_file()


def test_save_ttft_warmup_plot_comp_suffix_and_metadata(tmp_path: Path):
    ds = tmp_path / "mini.json"
    ds.write_text("[]")
    blend, full = _tiny_ttft_series()
    j_path, png_path = save_ttft_warmup_plot(
        str(ds),
        blend,
        full,
        stream_seed=42,
        skip_first=0,
        fifo_stats={"entries": 3},
        roll_window=2,
        cached_label="CompCache (composition-aware)",
        name_suffix="_comp",
    )
    assert j_path.endswith("mini_comp_ttft_warmup.json")
    data = json.loads(Path(j_path).read_text())
    assert data["ttft_blend_seconds"] == blend
    assert data["metadata"]["cached_label"] == "CompCache (composition-aware)"
    assert png_path and Path(png_path).is_file()


def test_fifo_warmup_default_suffix_matches_prior_naming(tmp_path: Path):
    ds = tmp_path / "mini.json"
    ds.write_text("[]")
    blend, full = _tiny_ttft_series(3)
    j_path, _ = save_ttft_warmup_plot(
        str(ds),
        blend,
        full,
        stream_seed=0,
        skip_first=0,
        fifo_stats={},
        roll_window=25,
    )
    assert Path(j_path).name == "mini_ttft_warmup.json"


def test_plot_ttft_warmup_json_script_renders_comp_label(tmp_path: Path):
    json_path = tmp_path / "sample_comp_ttft_warmup.json"
    blend, full = _tiny_ttft_series(4)
    json_path.write_text(
        json.dumps(
            {
                "ttft_blend_seconds": blend,
                "ttft_full_seconds": full,
                "metadata": {"cached_label": "CompCache (composition-aware)"},
            }
        )
    )
    out_png = tmp_path / "out.png"
    script = _REPO / "realistic_qa" / "scripts" / "plot_ttft_warmup_json.py"
    r = subprocess.run(
        [sys.executable, str(script), str(json_path), "-o", str(out_png), "-w", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert out_png.is_file()


def test_plot_compare_json_render_with_fifo_and_comp(tmp_path: Path):
    import importlib.util

    stem = "line"
    warmup = tmp_path / f"{stem}_ttft_warmup.json"
    comp_scores = tmp_path / f"{stem}_comp_scores.json"
    blend, full = _tiny_ttft_series(6)
    warmup.write_text(
        json.dumps(
            {
                "ttft_blend_seconds": blend,
                "ttft_full_seconds": full,
                "metadata": {},
            }
        )
    )
    comp_scores.write_text(
        json.dumps(
            {
                "ttft_blend": blend,
                "ttft_full": full,
            }
        )
    )
    mod_path = _REPO / "realistic_qa" / "scripts" / "plot_compare_json.py"
    spec = importlib.util.spec_from_file_location("plot_compare_json", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    out = tmp_path / "line_compare.png"
    written = mod.render(tmp_path, stem, out, roll_window=2)
    assert written == out
    assert out.is_file()


def test_histogram_empty_returns_no_paths(tmp_path: Path):
    ds = tmp_path / "mini.json"
    ds.write_text("[]")
    for suffix in ("", "_comp"):
        j, p = save_ttft_histogram(str(ds), [], [], name_suffix=suffix)
        assert j == "" and p is None
