"""Ensure standard_qa runner utilities are importable when executing from repo root."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STD_RUNNERS = _ROOT / "standard_qa" / "runners"
if _STD_RUNNERS.is_dir() and str(_STD_RUNNERS) not in sys.path:
    sys.path.insert(0, str(_STD_RUNNERS))
