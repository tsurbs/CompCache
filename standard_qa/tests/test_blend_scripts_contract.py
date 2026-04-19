"""Contract checks on ``standard_qa/runners/blend_*.py`` (full eval entrypoints, not ``blend.py``)."""
from __future__ import annotations

from pathlib import Path

_RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _blend_scripts():
    return sorted(_RUNNERS.glob("blend_*.py"))


def test_each_blend_script_uses_run_blend_eval_and_mistral():
    assert _blend_scripts(), f"expected blend_*.py under {_RUNNERS}"
    for path in _blend_scripts():
        if "_comp.py" in path.name:
            continue
        src = path.read_text(encoding="utf-8")
        assert "run_blend_eval" in src, f"{path.name} must call run_blend_eval"
        assert "mistralai/Mistral" in src, f"{path.name} must use Mistral (Modal / repo policy)"
        assert "standard_qa/inputs/" in src, f"{path.name} must use standard_qa/inputs datasets"


def test_each_blend_comp_script_uses_run_blend_eval_comp():
    comp_paths = sorted(_RUNNERS.glob("blend_*_comp.py"))
    assert comp_paths, "expected blend_*_comp.py scripts"
    for path in comp_paths:
        src = path.read_text(encoding="utf-8")
        assert "run_blend_eval_comp" in src, f"{path.name} must call run_blend_eval_comp"
        assert "mistralai/Mistral" in src
        assert "standard_qa/inputs/" in src


def test_blend_py_is_legacy_inline_not_run_blend_eval():
    legacy = _RUNNERS / "blend.py"
    assert legacy.is_file()
    src = legacy.read_text(encoding="utf-8")
    assert "run_blend_eval" not in src
