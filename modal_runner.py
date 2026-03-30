"""Run `example/blend_*.py` on Modal (Mistral checkpoints only, as in those scripts).

Target host profile (for scheduling / benchmarking metadata):
128 GiB system RAM, machine with 2× NVIDIA A40, ~1 TiB NVMe (measured ~4.8 GB/s seq.
read). The function pins one GPU via ``CUDA_VISIBLE_DEVICES=0`` and requests 128 GiB
RAM plus 1 TiB ephemeral scratch (HF cache, workspace).

``A40`` may not appear on every Modal workspace; set ``MODAL_GPU`` (e.g. ``L40S``)
if scheduling rejects the request.

**Image layer caching:** Only ``vllm_blend/`` is copied before the long ``pip install -e`` compile, then
``example/`` and ``inputs/`` are added in separate layers. Edits under ``example/``, ``inputs/``, or
this runner file therefore **do not** force a vLLM rebuild. Tune compile parallelism with
``MODAL_BUILD_MAX_JOBS`` (default ``4``) if the build worker OOMs or is killed during CMake/ninja.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
# Modal copies this module to ``/root/modal_runner.py`` for imports; sources live at ``/CompCache/…`` in the image.
_REPO_CONTAINER = Path("/CompCache")


def _repo_root() -> Path:
    if (_HERE / "vllm_blend").is_dir():
        return _HERE
    if (_REPO_CONTAINER / "vllm_blend").is_dir():
        return _REPO_CONTAINER
    return _HERE


REPO_ROOT = _repo_root()
_VLLM_SRC = REPO_ROOT / "vllm_blend"
_EXAMPLE_SRC = REPO_ROOT / "example"
_INPUTS_SRC = REPO_ROOT / "inputs"

# vLLM compile is heavy; Modal image builders can OOM if ninja uses too many jobs.
_BUILD_MAX_JOBS = os.environ.get("MODAL_BUILD_MAX_JOBS", "4")
# Blend scripts use paths like ``inputs/*.json``; those live at repo root, not under ``example/``.
_CONTAINER_REPO = "/CompCache"
BLEND_PATTERN = "blend_*.py"
# Matches checkpoints used in example/blend_*.py (Mistral LM, not Mixtral)
MISTRAL_ID_PREFIX = "mistralai/Mistral"

# 128 GiB RAM, 1 TiB ephemeral disk (Modal uses MiB for both request knobs).
_MEM_MIB = 128 * 1024
_DISK_MIB = 1024 * 1024

_GPU = os.environ.get("MODAL_GPU", "L40S")
if ":" not in _GPU:
    _GPU = f"{_GPU}:1"


def _blend_image() -> modal.Image:
    if not _VLLM_SRC.is_dir():
        raise FileNotFoundError(f"Missing vllm_blend: {_VLLM_SRC}")
    if not _EXAMPLE_SRC.is_dir():
        raise FileNotFoundError(f"Missing example: {_EXAMPLE_SRC}")
    if not _INPUTS_SRC.is_dir():
        raise FileNotFoundError(f"Missing inputs: {_INPUTS_SRC}")

    _copy_ignore = (".git", ".venv", "**/__pycache__", "*.pyc", ".DS_Store")
    # Local cmake/build trees should not bust the image hash or overwrite clean builds.
    _vllm_ignore = [
        *_copy_ignore,
        "build",
        "dist",
        "*.egg-info",
    ]

    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
            add_python="3.10",
        )
        .env(
            {
                "DEBIAN_FRONTEND": "noninteractive",
                "VLLM_TARGET_DEVICE": "cuda",
                "HF_HOME": "/cache/huggingface",
                "TRANSFORMERS_CACHE": "/cache/huggingface",
                "MAX_JOBS": "8",
                # Line-buffer print() from benchmarks / vLLM in ``modal run`` logs.
                "PYTHONUNBUFFERED": "1",
                # PyTorch 2.2.1 + vLLM 0.4.1 CMake fails on autodetected "9.0a"; pin arches.
                "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0",
            }
        )
        .apt_install("git", "build-essential", "ninja-build", "cmake", "curl")
        .pip_install(
            "pip",
            "setuptools",
            "wheel",
            "packaging",
            "numpy>=1.26,<2",
        )
        # Layer 1 — only vLLM sources; compile step reuses cache when example/notebooks/modal_runner change.
        .add_local_dir(
            _VLLM_SRC,
            remote_path="/CompCache/vllm_blend",
            copy=True,
            ignore=_vllm_ignore,
        )
        .run_commands(
            "bash -c 'set -euxo pipefail; echo \"[image] $(date -Is) starting vllm deps\"; python -V; nproc; echo MAX_JOBS=$MAX_JOBS'",
            "bash -c 'set -euxo pipefail; echo \"[image] $(date -Is) pip install rouge_score\"; python -m pip install rouge_score'",
            "bash -c 'set -euxo pipefail; echo \"[image] $(date -Is) pip install requirements-cuda\"; python -m pip install -r /CompCache/vllm_blend/requirements-cuda.txt'",
            "bash -c 'set -euxo pipefail; echo \"[image] $(date -Is) pip install numpy pin\"; python -m pip install \"numpy>=1.26,<2\"'",
            f"bash -c 'set -euxo pipefail; export MAX_JOBS={_BUILD_MAX_JOBS}; echo \"[image] $(date -Is) editable vllm build MAX_JOBS=$MAX_JOBS …\"; cd /CompCache/vllm_blend && python -m pip install --no-build-isolation -e .'",
            # Do not ``import vllm._C`` here: image builders have no GPU driver, so dlopen
            # fails with ``libcuda.so.1: cannot open shared object file``. Runtime containers
            # mount the driver and can load the extension.
            "bash -c 'echo \"[image] $(date -Is) vllm editable install finished; skipping _C import until runtime (no libcuda in build)\"'",
        )
        # Thin layers — frequent edits; do not invalidate the compile above.
        .add_local_dir(
            _EXAMPLE_SRC,
            remote_path="/CompCache/example",
            copy=True,
            ignore=_copy_ignore,
        )
        .add_local_dir(
            _INPUTS_SRC,
            remote_path="/CompCache/inputs",
            copy=True,
            ignore=_copy_ignore,
        )
    )


# Modal must import this module in the container; `include_source=False` breaks that.
app = modal.App("compcache-blend", image=_blend_image())

_FN = dict(
    gpu=_GPU,
    memory=_MEM_MIB,
    ephemeral_disk=_DISK_MIB,
    cpu=16.0,
    timeout=86400,
)


def _mistral_only(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    if MISTRAL_ID_PREFIX not in text:
        raise ValueError(f"{script.name} must use {MISTRAL_ID_PREFIX} …")
    if "Mixtral" in text or "mixtral" in text.lower():
        raise ValueError(f"{script.name} must not use Mixtral")
    lower = text.lower()
    for forbidden in ("meta-llama/", "qwen/", "microsoft/phi", "google/gemma"):
        if forbidden in lower:
            raise ValueError(f"{script.name} references a non-Mistral model ({forbidden})")


def _run_blend_script(path: Path) -> None:
    """Run one benchmark with unbuffered Python so logs stream to Modal."""
    print(f"[modal] running {path.name} …", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(path)],
        cwd=_CONTAINER_REPO,
        check=True,
    )


@app.function(**_FN)
def run_blend(script: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONPATH"] = "/CompCache/vllm_blend:/CompCache/example"

    ex_dir = Path("/CompCache/example")
    path = ex_dir / script
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.match(BLEND_PATTERN):
        raise ValueError(f"Not a blend script: {script}")
    _mistral_only(path)

    _run_blend_script(path)


@app.function(**_FN)
def run_all_blends() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONPATH"] = "/CompCache/vllm_blend:/CompCache/example"

    ex_dir = Path("/CompCache/example")
    paths = sorted(ex_dir.glob(BLEND_PATTERN))
    print(f"[modal] run_all_blends: {len(paths)} script(s)", flush=True)
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)


@app.local_entrypoint()
def main(script: str | None = None):
    # Without this, some Modal / CLI versions show little or no remote stdout locally.
    with modal.enable_output():
        if script:
            run_blend.remote(script)
        else:
            run_all_blends.remote()
