"""Env overrides for longctx blend runners (Modal sets these like ``REALISTIC_*``)."""
from __future__ import annotations

import os


def env_dataset(default: str) -> str:
    v = os.environ.get("LONGCTX_DATASET", "").strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return int(raw)
