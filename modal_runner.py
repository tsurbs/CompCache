from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
# Modal copies this module to ``/root/modal_runner.py`` for imports; sources live at ``/CompCache/…`` in the image.
_REPO_CONTAINER = Path("/CompCache")


def _repo_root() -> Path:
    """Repo root on the dev machine next to ``modal_runner.py``, or ``/CompCache`` in Modal layers."""
    if (_HERE / "vllm_blend").is_dir():
        return _HERE
    if (_REPO_CONTAINER / "vllm_blend").is_dir():
        return _REPO_CONTAINER
    # Dataset-builder image: only standard_qa + realistic_qa under /CompCache (no vLLM)
    if (_REPO_CONTAINER / "standard_qa").is_dir():
        return _REPO_CONTAINER
    return _HERE


REPO_ROOT = _repo_root()
_VLLM_SRC = REPO_ROOT / "vllm_blend"
_STANDARD_QA_SRC = REPO_ROOT / "standard_qa"
_RUNNERS_SRC = _STANDARD_QA_SRC / "runners"
_INPUTS_SRC = _STANDARD_QA_SRC / "inputs"
_REALISTIC_QA_SRC = REPO_ROOT / "realistic_qa"
_REALISTIC_RUNNERS_SRC = _REALISTIC_QA_SRC / "runners"
_REALISTIC_INPUTS_SRC = _REALISTIC_QA_SRC / "inputs"

# vLLM compile is very slow and memory intensive, limit number of jobs to avoid OOMs
_BUILD_MAX_JOBS = os.environ.get("MODAL_BUILD_MAX_JOBS", "4")
# Blend scripts use paths like standard_qa/inputs/*.json relative to repo root
_CONTAINER_REPO = "/CompCache"
_CONTAINER_STANDARD_QA = f"{_CONTAINER_REPO}/standard_qa"
_CONTAINER_RUNNERS = f"{_CONTAINER_STANDARD_QA}/runners"
_CONTAINER_INPUTS = f"{_CONTAINER_STANDARD_QA}/inputs"
_CONTAINER_REALISTIC = f"{_CONTAINER_REPO}/realistic_qa"
_CONTAINER_REALISTIC_RUNNERS = f"{_CONTAINER_REALISTIC}/runners"
_CONTAINER_REALISTIC_INPUTS = f"{_CONTAINER_REALISTIC}/inputs"
BLEND_PATTERN = "blend_*.py"
REALISTIC_SCRIPT = "blend_realistic.py"

MISTRAL_ID_PREFIX = "mistralai/Mistral"

# 256 GiB RAM (env-overridable), 1 TiB ephemeral disk.  The delta-memory
# sweep stacks a full-joint pair store, two delta stores, and a shared
# per-chunk FIFO — all pinned to host RAM — so we default high.
_MEM_MIB = int(os.environ.get("MODAL_MEM_MIB", str(256 * 1024)))
_DISK_MIB = 1024 * 1024

_GPU = os.environ.get("MODAL_GPU", "L40S")
if ":" not in _GPU:
    _GPU = f"{_GPU}:1"

# Lighter defaults for the dataset builder (embeddings only — no vLLM).
# Still needs GPU for SentenceTransformers / Faiss
_BUILDER_GPU = os.environ.get("MODAL_BUILDER_GPU", _GPU)
if ":" not in _BUILDER_GPU:
    _BUILDER_GPU = f"{_BUILDER_GPU}:1"
_BUILDER_MEM_MIB = int(os.environ.get("MODAL_BUILDER_MEM_MIB", str(64 * 1024)))
_BUILDER_DISK_MIB = int(os.environ.get("MODAL_BUILDER_DISK_MIB", str(512 * 1024)))
_BUILDER_CPU = float(os.environ.get("MODAL_BUILDER_CPU", "8"))


def _blend_image() -> modal.Image:
    if not _VLLM_SRC.is_dir():
        raise FileNotFoundError(f"Missing vllm_blend: {_VLLM_SRC}")
    if not _STANDARD_QA_SRC.is_dir():
        raise FileNotFoundError(f"Missing standard_qa: {_STANDARD_QA_SRC}")
    if not _RUNNERS_SRC.is_dir():
        raise FileNotFoundError(f"Missing standard_qa/runners: {_RUNNERS_SRC}")
    if not _INPUTS_SRC.is_dir():
        raise FileNotFoundError(f"Missing standard_qa/inputs: {_INPUTS_SRC}")
    if not _REALISTIC_QA_SRC.is_dir():
        raise FileNotFoundError(f"Missing realistic_qa: {_REALISTIC_QA_SRC}")
    if not _REALISTIC_RUNNERS_SRC.is_dir():
        raise FileNotFoundError(f"Missing realistic_qa/runners: {_REALISTIC_RUNNERS_SRC}")
    if not _REALISTIC_INPUTS_SRC.is_dir():
        raise FileNotFoundError(f"Missing realistic_qa/inputs: {_REALISTIC_INPUTS_SRC}")

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
            "matplotlib>=3.7,<4",
        )
        # Layer 1 — only vLLM sources; compile step reuses cache when standard_qa/modal_runner change.
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
        # Thin layer — full ``standard_qa/`` tree (runners + inputs); does not invalidate vLLM compile.
        .add_local_dir(
            _STANDARD_QA_SRC,
            remote_path="/CompCache/standard_qa",
            copy=True,
            ignore=_copy_ignore,
        )
        .add_local_dir(
            _REALISTIC_QA_SRC,
            remote_path="/CompCache/realistic_qa",
            copy=True,
            ignore=_copy_ignore,
        )
    )


def _dataset_builder_image() -> modal.Image:
    """CUDA + PyTorch + SentenceTransformers + Faiss (no vLLM) for ``build_extended_dataset.py``."""
    if not _STANDARD_QA_SRC.is_dir():
        raise FileNotFoundError(f"Missing standard_qa: {_STANDARD_QA_SRC}")
    if not _INPUTS_SRC.is_dir():
        raise FileNotFoundError(f"Missing standard_qa/inputs: {_INPUTS_SRC}")
    if not _REALISTIC_QA_SRC.is_dir():
        raise FileNotFoundError(f"Missing realistic_qa: {_REALISTIC_QA_SRC}")

    _copy_ignore = (".git", ".venv", "**/__pycache__", "*.pyc", ".DS_Store")
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
            add_python="3.10",
        )
        .env(
            {
                "DEBIAN_FRONTEND": "noninteractive",
                "HF_HOME": "/cache/huggingface",
                "TRANSFORMERS_CACHE": "/cache/huggingface",
                "PYTHONUNBUFFERED": "1",
            }
        )
        .pip_install("pip", "setuptools", "wheel", "numpy>=1.26,<2")
        .run_commands(
            "bash -c 'set -euxo pipefail; python -m pip install --no-cache-dir "
            "torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121'",
            "bash -c 'set -euxo pipefail; python -m pip install --no-cache-dir "
            "\"sentence-transformers>=2.2.0\" \"transformers>=4.36.0,<4.51.0\" "
            "\"langchain-text-splitters>=0.2.0\" \"openai>=1.0.0\" \"python-dotenv>=1.0.0\" "
            "\"tqdm>=4.66.0\" \"faiss-cpu>=1.7.4\"'",
        )
        .add_local_dir(
            _STANDARD_QA_SRC,
            remote_path="/CompCache/standard_qa",
            copy=True,
            ignore=_copy_ignore,
        )
        .add_local_dir(
            _REALISTIC_QA_SRC,
            remote_path="/CompCache/realistic_qa",
            copy=True,
            ignore=_copy_ignore,
        )
    )


# Modal imports this module inside every worker. Dataset-builder containers have no
# ``/CompCache/vllm_blend``, so ``_blend_image()`` must not run there unguarded.
_dataset_builder_image_singleton = _dataset_builder_image()
try:
    _blend_image_singleton = _blend_image()
except FileNotFoundError:
    _blend_image_singleton = _dataset_builder_image_singleton

app = modal.App("compcache-blend", image=_blend_image_singleton)

# Persistent artifact storage.  Every benchmark function mounts this
# Volume and mirrors its result JSONs / PNGs into it after each script
# completes (see ``_run_blend_script`` and per-runner persist hooks).
# Survives:
#   - container teardown (regular completion or SIGKILL),
#   - the local ``modal run`` process being killed/disconnected,
#   - mid-suite failures (whatever already finished is on the Volume).
# Pull a snapshot back to local disk with::
#
#     modal run modal_runner.py::pull_artifacts                # everything
#     modal run modal_runner.py::pull_artifacts --prefix standard_qa/
_ARTIFACTS_VOLUME = modal.Volume.from_name(
    "compcache-artifacts", create_if_missing=True,
)
_ARTIFACTS_VOLUME_MOUNT = "/artifacts"

_FN = dict(
    gpu=_GPU,
    memory=_MEM_MIB,
    ephemeral_disk=_DISK_MIB,
    cpu=16.0,
    timeout=86400,
    volumes={_ARTIFACTS_VOLUME_MOUNT: _ARTIFACTS_VOLUME},
)

_FN_BUILDER = dict(
    gpu=_BUILDER_GPU,
    memory=_BUILDER_MEM_MIB,
    ephemeral_disk=_BUILDER_DISK_MIB,
    cpu=_BUILDER_CPU,
    timeout=86400,
)


# Fixed name so Modal’s local and container imports resolve the same ``Secret`` graph node.
_DEFAULT_COMPCACHE_SECRET = "compcache-dotenv"

_WORKSPACE_SECRETS: list[modal.Secret] = [
    modal.Secret.from_name(_DEFAULT_COMPCACHE_SECRET),
]


def _fn_with_workspace_secrets(fn_kwargs: dict) -> dict:
    out = dict(fn_kwargs)
    prev = list(out.get("secrets") or [])
    out["secrets"] = prev + _WORKSPACE_SECRETS
    return out


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


def _maybe_pin_cuda_visible_device0() -> None:
    """Modal usually sets CUDA_VISIBLE_DEVICES; only override when explicitly asked."""
    if os.environ.get("MODAL_PIN_CUDA_0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _sys_path_prepend_for_vllm_and_runners() -> None:
    for p in (
        f"{_CONTAINER_REPO}/vllm_blend",
        _CONTAINER_RUNNERS,
        _CONTAINER_REALISTIC_RUNNERS,
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def _invoke_blend_realistic_main() -> None:
    """Run eval in-process so CUDA initializes once (avoids subprocess + bad GPU state quirks)."""
    prev = os.getcwd()
    _sys_path_prepend_for_vllm_and_runners()
    try:
        os.chdir(_CONTAINER_REPO)
        br = importlib.import_module("blend_realistic")
        br.main()
    finally:
        os.chdir(prev)


def _run_blend_script(path: Path) -> None:
    """Run one benchmark with unbuffered Python so logs stream to Modal.

    Always persists whatever artifacts exist on disk to the durable
    Volume in a ``finally``, so a crash or SIGKILL on script N still
    leaves us with the outputs of scripts 1..N-1 (and any partial
    checkpoint that script N managed to flush).
    """
    print(f"[modal] running {path.name} …", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=_CONTAINER_REPO,
            check=True,
        )
    finally:
        try:
            _persist_artifacts_to_volume()
        except Exception as e:  # pragma: no cover - best effort
            print(f"[modal] volume persist failed after {path.name}: {e!r}", flush=True)


def _persist_artifacts_to_volume() -> int:
    """Mirror everything ``_collect_result_jsons`` would gather into the
    persistent Volume and ``commit()`` so the data survives container
    teardown.  Idempotent — re-runs overwrite identically named files
    with the latest bytes.  Returns the number of files written.
    """
    artifacts = _collect_result_jsons()
    if not artifacts:
        return 0
    base = Path(_ARTIFACTS_VOLUME_MOUNT)
    n = 0
    for rel, data in artifacts.items():
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        n += 1
    _ARTIFACTS_VOLUME.commit()
    print(f"[modal] persisted {n} artifact(s) to volume '{_ARTIFACTS_VOLUME.name if hasattr(_ARTIFACTS_VOLUME, 'name') else 'compcache-artifacts'}'", flush=True)
    return n


def _collect_result_jsons() -> dict[str, bytes]:
    """Gather all analysis JSON files produced during the run."""
    results = {}
    for inputs_dir, prefix in (
        (Path(_CONTAINER_INPUTS), "standard_qa/inputs/"),
        (Path(_CONTAINER_REALISTIC_INPUTS), "realistic_qa/inputs/"),
    ):
        for pattern in (
            "*_coretrieval.json",
            "*_comp_coretrieval.json",
            "*_3way*_coretrieval.json",
            "*_scores.json",
            "*_comp_scores.json",
            "*_3way*_scores.json",
            "*_synthetic_zipf.json",
            "*_ttft_warmup.json",
            "*_ttft_warmup.png",
            "*_ttft_hist.json",
            "*_ttft_hist.png",
            "*_3way*_ttft_warmup.json",
            "*_3way*_ttft_warmup.png",
            "*_3way*_ttft_hist.json",
            "*_3way*_ttft_hist.png",
            "*_recomp_sweep*_scores.json",
            # Sparse-delta CompCache variants (blend_*_comp_delta.py).
            "*_comp_delta_coretrieval.json",
            "*_comp_delta_scores.json",
            "*_comp_delta_ttft_warmup.json",
            "*_comp_delta_ttft_warmup.png",
            "*_comp_delta_ttft_hist.json",
            "*_comp_delta_ttft_hist.png",
            "*_comp_delta_vs_full.png",
            # Memory-savings sweep (Test 1).
            "*_delta_memory_scores.json",
            "*_delta_memory_timeseries.png",
            "*_delta_memory_tradeoff.png",
            "*_delta_memory_ttft.png",
            # Popularity / equal-bytes budget (Test 2).
            "*_budget_scores.json",
            "*_budget_main.png",
            "*_budget_ttft_hist.png",
        ):
            for p in sorted(inputs_dir.glob(pattern)):
                rel = prefix + p.name
                results[rel] = p.read_bytes()
                print(f"[modal] collected {rel} ({len(results[rel])} bytes)", flush=True)
    return results


def _save_results_locally(artifacts: dict[str, bytes]) -> None:
    """Write downloaded result JSONs under ``standard_qa/inputs/`` or ``realistic_qa/inputs/``."""
    for name, data in artifacts.items():
        dest = REPO_ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"[local] saved {dest}")


def _read_artifacts_from_volume(prefix: str = "") -> dict[str, bytes]:
    """Read every file under the persistent Volume mount, optionally
    filtered by a leading path ``prefix`` (e.g. ``"standard_qa/"``).

    Returns the same ``{repo-relative-path: bytes}`` shape as
    ``_collect_result_jsons`` so the result can be fed straight into
    ``_save_results_locally``.
    """
    base = Path(_ARTIFACTS_VOLUME_MOUNT)
    out: dict[str, bytes] = {}
    if not base.exists():
        print(f"[modal] volume mount {base} does not exist", flush=True)
        return out
    # Make sure we see writes from any concurrent container.
    try:
        _ARTIFACTS_VOLUME.reload()
    except Exception as e:  # pragma: no cover - best effort
        print(f"[modal] volume reload failed: {e!r}", flush=True)
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(base))
        if prefix and not rel.startswith(prefix):
            continue
        out[rel] = p.read_bytes()
    print(
        f"[modal] read {len(out)} file(s) from volume "
        f"(prefix={prefix!r})",
        flush=True,
    )
    return out


# CPU-only, no GPU — these just read a mounted Volume, so we must not
# pay for an H100 every time we poll progress overnight.  Ephemeral
# disk omitted — Modal enforces a 512 GiB minimum there, and we don't
# need scratch space for a Volume read.
_FN_VOLUME_CPU = dict(
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={_ARTIFACTS_VOLUME_MOUNT: _ARTIFACTS_VOLUME},
)


@app.function(**_fn_with_workspace_secrets(_FN_VOLUME_CPU))
def pull_artifacts_from_volume(prefix: str = "") -> dict[str, bytes]:
    """Snapshot the persistent artifact Volume into an in-memory dict.

    Use the ``pull_artifacts`` local entrypoint to also write them to
    your local disk.  ``prefix`` (optional) filters by Volume-relative
    path, e.g. ``"standard_qa/"`` or ``"realistic_qa/inputs/foo_"``.
    """
    return _read_artifacts_from_volume(prefix=prefix)


@app.function(**_fn_with_workspace_secrets(_FN_VOLUME_CPU))
def list_artifacts_in_volume(prefix: str = "") -> list[tuple[str, int]]:
    """Cheap listing (path, size_bytes) of what's currently on the Volume.

    Useful to verify a long-running detached job is actually persisting
    things without paying to download them all.
    """
    base = Path(_ARTIFACTS_VOLUME_MOUNT)
    if not base.exists():
        return []
    try:
        _ARTIFACTS_VOLUME.reload()
    except Exception:
        pass
    rows: list[tuple[str, int]] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(base))
        if prefix and not rel.startswith(prefix):
            continue
        rows.append((rel, p.stat().st_size))
    return rows


def _collect_builder_outputs(mode: str, out: str, out_samsum: str) -> dict[str, bytes]:
    results: dict[str, bytes] = {}
    rels: list[str] = []
    if mode in ("musique_wikimqa", "both"):
        rels.append(out)
    if mode in ("samsum", "both"):
        rels.append(out_samsum)
    for rel in rels:
        p = Path(_CONTAINER_REPO) / rel
        if p.is_file():
            results[rel] = p.read_bytes()
            print(f"[modal] collected {rel} ({len(results[rel])} bytes)", flush=True)
        else:
            print(f"[modal] warning: missing output {p}", flush=True)
    return results


@app.function(
    **_fn_with_workspace_secrets(
        {**_FN_BUILDER, "image": _dataset_builder_image_singleton},
    )
)
def run_build_extended_dataset(
    mode: str = "musique_wikimqa",
    n_per_set: int = 750,
    seed: int = 42,
    out: str = "realistic_qa/inputs/extended_cacheblend.json",
    out_samsum: str = "realistic_qa/inputs/extended_samsum.json",
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    st_device: str = "cuda",
    embed_batch_size: int = 128,
    chat_model: str = "gpt-4.1-mini",
    gateway: str = "",
    top_k: int = 6,
    shuffle_seed: int = 34,
    samsum_n: int = 1500,
    splitter_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
) -> dict[str, bytes]:
    """GPU-accelerated extended dataset build (SentenceTransformers + Faiss)."""
    _maybe_pin_cuda_visible_device0()
    script = Path(_CONTAINER_REPO) / "realistic_qa" / "scripts" / "build_extended_dataset.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(script),
        "--mode",
        mode,
        "--out",
        out,
        "--out-samsum",
        out_samsum,
        "--seed",
        str(seed),
        "--n-per-set",
        str(n_per_set),
        "--samsum-n",
        str(samsum_n),
        "--st-model",
        st_model,
        "--st-device",
        st_device,
        "--embed-batch-size",
        str(embed_batch_size),
        "--chat-model",
        chat_model,
        "--splitter-model",
        splitter_model,
        "--top-k",
        str(top_k),
        "--shuffle-seed",
        str(shuffle_seed),
    ]
    gw = gateway.strip()
    if gw:
        cmd.extend(["--gateway", gw])

    print("[modal] build_extended_dataset argv:", cmd[3:], flush=True)
    subprocess.run(cmd, cwd=_CONTAINER_REPO, check=True, env=os.environ.copy())
    return _collect_builder_outputs(mode, out, out_samsum)


@app.function(**_fn_with_workspace_secrets(_FN))
def run_blend(script: str) -> dict[str, bytes]:
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )

    ex_dir = Path(_CONTAINER_RUNNERS)
    path = ex_dir / script
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.match(BLEND_PATTERN):
        raise ValueError(f"Not a blend script: {script}")
    _mistral_only(path)

    _run_blend_script(path)
    return _collect_result_jsons()


def _realistic_py_path() -> Path:
    return Path(_CONTAINER_REALISTIC_RUNNERS) / REALISTIC_SCRIPT


@app.function(**_fn_with_workspace_secrets(_FN))
def run_realistic_blend(
    dataset: str = "realistic_qa/inputs/extended_tiny.json",
    fifo_max: int = 10_000,
    skip_first: int = 0,
    max_ctx_len: int = 8192,
    max_model_len: int = 0,
    gpu_memory_utilization: float = 0.45,
    mode: str = "fifo",
    pair_store_capacity: int = 256,
    pair_store_kind: str = "full",
    delta_top_k_ratio: float = 0.1,
    promotion_threshold: int = 10,
    promote_sync: bool = False,
) -> dict[str, bytes]:
    """CacheBlend evaluation on a realistic_qa JSON (see ``blend_realistic.py``).

    ``mode=fifo`` reproduces the baseline FIFO path; ``mode=comp`` enables
    composition-aware pair KV caching (Proposal §3.1/§3.2): a ``PairKVStore``
    + async ``PromotionWorker`` that materializes joint KV for frequently
    co-retrieved document pairs. ``pair_store_capacity`` is the max number
    of promoted pairs held at once; ``promotion_threshold`` is the
    co-retrieval count at which a pair gets enqueued for async promotion;
    ``promote_sync`` forces promotion on the triggering query
    (single-threaded, deterministic microbench mode).
    ``pair_store_kind="full"`` (§3.1) stores the whole joint KV;
    ``pair_store_kind="delta"`` (§3.2) stores only the sparse
    ``Δ = joint - cat(individual_a, individual_b)`` retaining
    ``delta_top_k_ratio`` of positions per layer (lower memory, tiny
    reconstruction noise).

    ``max_ctx_len`` trims passage chunks (middle-out) so long multi-hop prompts fit
    on a ~48GB GPU; use 0 to disable. ``max_model_len`` caps vLLM context (scheduler
    + KV); 0 lets ``blend_realistic`` derive ``max_ctx_len + 512`` when
    ``max_ctx_len`` is set, else Hugging Face max. Lower ``gpu_memory_utilization``
    leaves VRAM for FIFO-cloned KV tensors alongside vLLM's pool.
    """
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )
    os.environ["REALISTIC_DATASET"] = dataset
    os.environ["REALISTIC_FIFO_MAX"] = str(fifo_max)
    os.environ["REALISTIC_SKIP_FIRST"] = str(skip_first)
    os.environ["REALISTIC_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
    os.environ["REALISTIC_MODE"] = mode
    os.environ["REALISTIC_PAIR_STORE_CAP"] = str(pair_store_capacity)
    os.environ["REALISTIC_PAIR_STORE_KIND"] = pair_store_kind
    os.environ["REALISTIC_DELTA_TOP_K_RATIO"] = str(delta_top_k_ratio)
    os.environ["REALISTIC_PROMOTION_THRESHOLD"] = str(promotion_threshold)
    os.environ["REALISTIC_PROMOTE_SYNC"] = "1" if promote_sync else "0"
    if max_ctx_len > 0:
        os.environ["REALISTIC_MAX_CTX_LEN"] = str(max_ctx_len)
    else:
        os.environ.pop("REALISTIC_MAX_CTX_LEN", None)
    if max_model_len > 0:
        os.environ["REALISTIC_MAX_MODEL_LEN"] = str(max_model_len)
    else:
        os.environ.pop("REALISTIC_MAX_MODEL_LEN", None)

    path = _realistic_py_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    _mistral_only(path)

    print(
        f"[modal] realistic QA: dataset={dataset} mode={mode} fifo_max={fifo_max} "
        f"skip_first={skip_first} max_ctx_len={max_ctx_len} "
        f"max_model_len={max_model_len or '(auto or HF)'} "
        f"gpu_memory_utilization={gpu_memory_utilization} "
        f"pair_store_cap={pair_store_capacity} pair_store_kind={pair_store_kind} "
        f"delta_top_k_ratio={delta_top_k_ratio} "
        f"promotion_threshold={promotion_threshold} "
        f"promote_sync={promote_sync}",
        flush=True,
    )
    try:
        _invoke_blend_realistic_main()
    finally:
        try:
            _persist_artifacts_to_volume()
        except Exception as e:  # pragma: no cover - best effort
            print(f"[modal] volume persist failed after run_realistic_blend: {e!r}", flush=True)
    return _collect_result_jsons()


@app.function(**_fn_with_workspace_secrets(_FN))
def run_all_blends() -> dict[str, bytes]:
    _maybe_pin_cuda_visible_device0()
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )

    ex_dir = Path(_CONTAINER_RUNNERS)
    paths = sorted(
        p for p in ex_dir.glob(BLEND_PATTERN)
        if "_comp.py" not in p.name
        and "_comp_delta.py" not in p.name
        and "_3way.py" not in p.name
        and "_sweep.py" not in p.name
    )
    print(f"[modal] run_all_blends: {len(paths)} script(s)", flush=True)
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)
    return _collect_result_jsons()


@app.function(**_fn_with_workspace_secrets(_FN))
def run_all_blends_comp() -> dict[str, bytes]:
    """Run every ``standard_qa/runners/blend_*_comp.py`` (CompCache + TTFT plots)."""
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )

    ex_dir = Path(_CONTAINER_RUNNERS)
    paths = sorted(ex_dir.glob("blend_*_comp.py"))
    print(f"[modal] run_all_blends_comp: {len(paths)} script(s)", flush=True)
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)
    return _collect_result_jsons()


@app.local_entrypoint()
def standard_comp():
    """Run every standard CompCache benchmark (``blend_*_comp.py``); downloads TTFT + coretrieval like FIFO."""
    with modal.enable_output():
        artifacts = run_all_blends_comp.remote()
    _save_results_locally(artifacts)


@app.function(**_fn_with_workspace_secrets(_FN))
def run_all_blends_comp_delta(
    delta_top_k_ratio: float | None = None,
    artifact_suffix: str | None = None,
) -> dict[str, bytes]:
    """Run every ``standard_qa/runners/blend_*_comp_delta.py``.

    Identical pipeline to :func:`run_all_blends_comp` but each driver uses
    the sparse-delta pair store instead of the full-joint one.  Artifacts
    land under ``*_comp_delta_*`` so they coexist with a prior
    ``standard_comp`` run on the same dataset.  Optional knobs override
    the per-script defaults via env:

    - ``delta_top_k_ratio``: sparsity for ``SparseDeltaPairStore`` (default 0.10).
    - ``artifact_suffix``: override the ``comp_delta`` filename suffix.
    """
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )
    if delta_top_k_ratio is not None:
        os.environ["STANDARD_COMP_DELTA_TOP_K_RATIO"] = str(delta_top_k_ratio)
    if artifact_suffix:
        os.environ["STANDARD_COMP_ARTIFACT_SUFFIX"] = artifact_suffix

    ex_dir = Path(_CONTAINER_RUNNERS)
    paths = sorted(ex_dir.glob("blend_*_comp_delta.py"))
    print(
        f"[modal] run_all_blends_comp_delta: {len(paths)} script(s) "
        f"top_k_ratio={delta_top_k_ratio} suffix={artifact_suffix!r}",
        flush=True,
    )
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)
    return _collect_result_jsons()


@app.local_entrypoint()
def standard_comp_delta(
    delta_top_k_ratio: float | None = None,
    artifact_suffix: str | None = None,
):
    """Run every delta-store CompCache benchmark (``blend_*_comp_delta.py``).

    Pair with :func:`standard_comp` (the Full-Joint baseline) to compare
    F1 / TTFT / memory.  Both write to the same ``standard_qa/inputs/``
    directory under disjoint filename suffixes (``*_comp_*`` vs
    ``*_comp_delta_*``).  If a matching ``*_comp_scores.json`` exists
    locally for a dataset, this entrypoint also renders the
    ``*_comp_delta_vs_full.png`` comparison plot on the local machine
    after the Modal job completes — no extra GPU time needed.
    """
    with modal.enable_output():
        artifacts = run_all_blends_comp_delta.remote(
            delta_top_k_ratio=delta_top_k_ratio,
            artifact_suffix=artifact_suffix,
        )
    _save_results_locally(artifacts)

    # Post-hoc: for every *_comp_delta_scores.json we just saved, look
    # for a sibling *_comp_scores.json and render the combined plot.
    # This runs on the user's machine (no vLLM), so it's effectively
    # free even when the Modal job took minutes.
    inputs_dir = REPO_ROOT / "standard_qa" / "inputs"
    import sys

    sys.path.insert(0, str(REPO_ROOT / "standard_qa" / "runners"))
    try:
        from plot_delta_vs_full import plot_comparison  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"[local] skipping comparison plots ({e!r})")
        return
    for delta_scores in sorted(inputs_dir.glob("*_comp_delta_scores.json")):
        stem = delta_scores.stem
        if not stem.endswith("_comp_delta_scores"):
            continue
        base = stem[: -len("_comp_delta_scores")]
        full_scores = inputs_dir / f"{base}_comp_scores.json"
        if not full_scores.exists():
            print(
                f"[local] comparison plot skipped for {base}: "
                f"no {full_scores.name} alongside"
            )
            continue
        out = inputs_dir / f"{base}_comp_delta_vs_full.png"
        try:
            plot_comparison(str(full_scores), str(delta_scores), str(out))
        except Exception as e:  # pragma: no cover
            print(f"[local] plot_comparison failed for {base}: {e!r}")


@app.function(**_fn_with_workspace_secrets(_FN))
def run_all_blends_3way(
    recomp_ratio: float | None = None,
    pair_recomp_ratio: float | None = None,
    output_tag: str = "",
) -> dict[str, bytes]:
    """Run every ``standard_qa/runners/blend_*_3way.py`` (Full vs CompCache-single vs CompCache+pairs).

    Optional knobs (propagated via env to the standard_qa 3-way wrapper):

    - ``recomp_ratio``: overrides the per-script ``recomp_ratio`` for single-chunk.
    - ``pair_recomp_ratio``: overrides the per-script ``pair_recomp_ratio`` for +pairs.
    - ``output_tag``: if non-empty, artifacts are written as ``*_3way_<tag>_*`` so
      prior runs under ``*_3way_*`` are preserved.
    """
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )
    if recomp_ratio is not None:
        os.environ["STANDARD_COMP_RECOMP_RATIO"] = str(recomp_ratio)
    if pair_recomp_ratio is not None:
        os.environ["STANDARD_COMP_PAIR_RECOMP_RATIO"] = str(pair_recomp_ratio)
    if output_tag:
        os.environ["THREE_WAY_OUTPUT_TAG"] = output_tag

    ex_dir = Path(_CONTAINER_RUNNERS)
    paths = sorted(ex_dir.glob("blend_*_3way.py"))
    print(
        f"[modal] run_all_blends_3way: {len(paths)} script(s) "
        f"recomp={recomp_ratio} pair_recomp={pair_recomp_ratio} tag={output_tag!r}",
        flush=True,
    )
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)
    return _collect_result_jsons()


@app.local_entrypoint()
def standard_3way(
    recomp_ratio: float | None = None,
    pair_recomp_ratio: float | None = None,
    output_tag: str = "",
):
    """Run every standard 3-way benchmark (``blend_*_3way.py``); downloads ``*_3way_*`` artifacts.

    Pass ``--output-tag r018`` to preserve prior ``*_3way_*`` artifacts; new run is
    written under ``*_3way_r018_*``.
    """
    with modal.enable_output():
        artifacts = run_all_blends_3way.remote(
            recomp_ratio=recomp_ratio,
            pair_recomp_ratio=pair_recomp_ratio,
            output_tag=output_tag,
        )
    _save_results_locally(artifacts)


@app.function(**_fn_with_workspace_secrets(_FN))
def run_recomp_sweep(output_tag: str = "") -> dict[str, bytes]:
    """Run the HotpotQA recomputation-ratio sweep (``blend_hotpotqa_sweep.py``)."""
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )
    if output_tag:
        os.environ["RECOMP_SWEEP_OUTPUT_TAG"] = output_tag

    path = Path(_CONTAINER_RUNNERS) / "blend_hotpotqa_sweep.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    _mistral_only(path)
    print(f"[modal] run_recomp_sweep: {path.name} tag={output_tag!r}", flush=True)
    _run_blend_script(path)
    return _collect_result_jsons()


@app.local_entrypoint()
def recomp_sweep(output_tag: str = ""):
    """HotpotQA recomputation-ratio sweep; downloads ``*_recomp_sweep_*`` artifacts."""
    with modal.enable_output():
        artifacts = run_recomp_sweep.remote(output_tag=output_tag)
    _save_results_locally(artifacts)


@app.local_entrypoint()
def main(script: str | None = None, realistic: bool = False):
    """Default entry: all standard blends, or one ``blend_*.py``, or realistic if ``--realistic``."""
    with modal.enable_output():
        if realistic:
            artifacts = run_realistic_blend.remote()
        elif script:
            artifacts = run_blend.remote(script)
        else:
            artifacts = run_all_blends.remote()
    _save_results_locally(artifacts)


@app.local_entrypoint()
def realistic(
    dataset: str = "realistic_qa/inputs/extended_tiny.json",
    fifo_max: int = 10_000,
    skip_first: int = 0,
    max_ctx_len: int = 8192,
    max_model_len: int = 0,
    gpu_memory_utilization: float = 0.45,
    mode: str = "fifo",
    pair_store_capacity: int = 256,
    pair_store_kind: str = "full",
    delta_top_k_ratio: float = 0.1,
    promotion_threshold: int = 10,
    promote_sync: bool = False,
):
    """Run extended CacheBlend on Modal (FIFO, composition-aware, or 3-way).

    Default is the committed smoke fixture. For the full 6K benchmark, build locally then::

        modal run modal_runner.py::realistic --dataset realistic_qa/inputs/extended_cacheblend.json

    On ~48GB GPUs, keep default ``max_ctx_len`` (8192) or lower; use ``--max-ctx-len 0``
    only if you have enough VRAM for full Mistral-32k-style profiles.

    ``mode=3way`` runs Full Prefill / CompCache (single-chunk) / CompCache (+pairs)
    back-to-back per query against independent FIFO state per method, writing
    ``*_3way_scores.json`` and ``*_3way_ttft_*`` plots.

    Examples::

        modal run modal_runner.py::realistic
        modal run modal_runner.py::realistic --skip-first 1000
        modal run modal_runner.py::realistic --mode comp --pair-store-capacity 256 \
            --promotion-threshold 10
        modal run modal_runner.py::realistic --mode comp --promote-sync    # microbench
        modal run modal_runner.py::realistic --mode 3way --dataset realistic_qa/inputs/extended_cacheblend.json
    """
    with modal.enable_output():
        artifacts = run_realistic_blend.remote(
            dataset=dataset,
            fifo_max=fifo_max,
            skip_first=skip_first,
            max_ctx_len=max_ctx_len,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            mode=mode,
            pair_store_capacity=pair_store_capacity,
            pair_store_kind=pair_store_kind,
            delta_top_k_ratio=delta_top_k_ratio,
            promotion_threshold=promotion_threshold,
            promote_sync=promote_sync,
        )
    _save_results_locally(artifacts)


@app.local_entrypoint()
def delta_memory(
    dataset: str = "realistic_qa/inputs/extended_cacheblend.json",
    # Entry cap — kept high so the byte budget is what actually shapes
    # each store.  Set via REALISTIC_PAIR_STORE_CAP in the runner.
    pair_store_capacity: int = 4096,
    # Total bytes each pair store may hold.  Full-Joint and Sparse-Δ
    # share the same budget; Δ configs naturally fit ~1/top_k_ratio
    # more entries, which is the memory-savings story the benchmark
    # demonstrates.  Defaults to 2 GiB.  Set to 0 to disable and fall
    # back to entry-count-only eviction.
    bytes_budget_gib: float = 2.0,
    # Lowered from 10 → 0 so every co-retrieved pair is admitted on
    # first sight (no waiting period).  At threshold=10 on the 1400-
    # query extended_cacheblend stream only 17 of 10,314 unique pairs
    # ever crossed the bar and the cache stayed empty until query ~1000.
    promotion_threshold: int = 0,
    # Selective-recomputation ratio passed into vLLM's cache_fuse_metadata.
    # 0.18 matches the standard_qa comp/comp_delta sweeps so F1 numbers
    # stay comparable across benchmarks.  Set to 0 to fall back to
    # vLLM's built-in 0.16 default.
    recomp_ratio: float = 0.18,
    configs: str = "full,delta_r0.50,delta_r0.10",
    skip_first: int = 0,
    max_ctx_len: int = 8192,
    max_model_len: int = 0,
    # Three configs each clone KV tensors onto GPU inside assemble; we need
    # more headroom than the 0.45 default so the vLLM block pool does not
    # crowd our per-query working set (seen as CUDA OOM at ~830/1400).
    # 0.30 was too low — weights alone take 13.5 GB on a 44 GB L40S so
    # vLLM ended up with 0 KV blocks.  0.38 gives vLLM ~3 GB pool
    # (~1200 blocks, plenty for single-seq generate) and keeps ~11 GB
    # free for our per-query GPU clones + torch allocator workspace.
    gpu_memory_utilization: float = 0.38,
):
    """Test 1 — memory-savings sweep on a single dataset.

    Reuses ``run_realistic_blend`` but forces ``mode=delta_memory`` and
    threads the config sweep through ``DELTA_MEMORY_CONFIGS``.  Produces
    ``*_delta_memory_timeseries.png`` (memory MB vs query, one line per
    config), ``*_delta_memory_tradeoff.png`` (peak MB vs mean F1 Pareto),
    and ``*_delta_memory_ttft.png``.
    """
    extra_env = {"DELTA_MEMORY_CONFIGS": configs}
    if bytes_budget_gib > 0:
        extra_env["DELTA_MEMORY_BYTES_BUDGET"] = str(
            int(bytes_budget_gib * (1024 ** 3))
        )
    if recomp_ratio > 0:
        extra_env["REALISTIC_RECOMP_RATIO"] = f"{recomp_ratio:g}"
    with modal.enable_output():
        artifacts = _run_realistic_with_extra_env.remote(
            dataset=dataset,
            mode="delta_memory",
            pair_store_capacity=pair_store_capacity,
            promotion_threshold=promotion_threshold,
            skip_first=skip_first,
            max_ctx_len=max_ctx_len,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_env=extra_env,
        )
    _save_results_locally(artifacts)


@app.local_entrypoint()
def budget(
    dataset: str = "realistic_qa/inputs/extended_cacheblend.json",
    cap_full: int = 256,
    delta_top_k_ratio: float = 0.10,
    cap_delta: int = 0,
    promotion_threshold: int = 10,
    skip_first: int = 0,
    max_ctx_len: int = 8192,
    max_model_len: int = 0,
    gpu_memory_utilization: float = 0.45,
):
    """Test 2 — popularity / equal-bytes budget revamped realistic eval.

    Four methods per query: Full Prefill / CompCache-single / CompCache +
    Full-LFU pairs (cap=``cap_full``) / CompCache + Δ-sparse-LFU pairs
    at the matched bytes budget (cap ≈ cap_full / ratio).  ``cap_delta=0``
    (default) lets the runner derive ``cap_delta`` from ``cap_full /
    delta_top_k_ratio``; pass an explicit int to override.
    """
    with modal.enable_output():
        artifacts = _run_realistic_with_extra_env.remote(
            dataset=dataset,
            mode="budget",
            skip_first=skip_first,
            max_ctx_len=max_ctx_len,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            extra_env={
                "REALISTIC_CAP_FULL": str(cap_full),
                "REALISTIC_DELTA_TOP_K_RATIO": str(delta_top_k_ratio),
                "REALISTIC_CAP_DELTA": str(cap_delta) if cap_delta > 0 else "",
                "REALISTIC_PROMOTION_THRESHOLD": str(promotion_threshold),
            },
        )
    _save_results_locally(artifacts)


@app.function(**_fn_with_workspace_secrets(_FN))
def _run_realistic_with_extra_env(
    dataset: str,
    mode: str,
    *,
    pair_store_capacity: int = 256,
    promotion_threshold: int = 10,
    skip_first: int = 0,
    max_ctx_len: int = 8192,
    max_model_len: int = 0,
    gpu_memory_utilization: float = 0.45,
    extra_env: dict | None = None,
) -> dict[str, bytes]:
    """Shared Modal function for the two new test modes.

    Sets the usual ``REALISTIC_*`` env vars plus any ``extra_env`` overrides
    (e.g. ``DELTA_MEMORY_CONFIGS`` or ``REALISTIC_CAP_FULL``) and invokes
    ``blend_realistic.main`` in-process, then collects artifacts.
    """
    _maybe_pin_cuda_visible_device0()
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc
    os.environ["PYTHONPATH"] = (
        f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}:{_CONTAINER_REALISTIC_RUNNERS}"
    )
    os.environ["REALISTIC_DATASET"] = dataset
    os.environ["REALISTIC_SKIP_FIRST"] = str(skip_first)
    os.environ["REALISTIC_GPU_MEMORY_UTILIZATION"] = str(gpu_memory_utilization)
    os.environ["REALISTIC_MODE"] = mode
    os.environ["REALISTIC_PAIR_STORE_CAP"] = str(pair_store_capacity)
    os.environ["REALISTIC_PROMOTION_THRESHOLD"] = str(promotion_threshold)
    if max_ctx_len > 0:
        os.environ["REALISTIC_MAX_CTX_LEN"] = str(max_ctx_len)
    else:
        os.environ.pop("REALISTIC_MAX_CTX_LEN", None)
    if max_model_len > 0:
        os.environ["REALISTIC_MAX_MODEL_LEN"] = str(max_model_len)
    else:
        os.environ.pop("REALISTIC_MAX_MODEL_LEN", None)
    for k, v in (extra_env or {}).items():
        if v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    path = _realistic_py_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    _mistral_only(path)
    print(
        f"[modal] {mode}: dataset={dataset} "
        f"extra_env={sorted((extra_env or {}).items())}",
        flush=True,
    )
    try:
        _invoke_blend_realistic_main()
    finally:
        try:
            _persist_artifacts_to_volume()
        except Exception as e:  # pragma: no cover - best effort
            print(f"[modal] volume persist failed after {mode}: {e!r}", flush=True)
    return _collect_result_jsons()


@app.local_entrypoint()
def pull_artifacts(prefix: str = ""):
    """Download every artifact persisted to the ``compcache-artifacts``
    Modal Volume by prior runs and write them under the matching local
    ``standard_qa/inputs/`` / ``realistic_qa/inputs/`` directory.

    Safe to invoke any time — even while another detached job is still
    running — because the writers commit after every script.  Existing
    local files are overwritten with the latest Volume copy.

    Examples::

        modal run modal_runner.py::pull_artifacts
        modal run modal_runner.py::pull_artifacts --prefix standard_qa/
        modal run modal_runner.py::pull_artifacts \\
            --prefix standard_qa/inputs/hotpotqa_s_comp_delta
    """
    with modal.enable_output():
        artifacts = pull_artifacts_from_volume.remote(prefix=prefix)
    if not artifacts:
        print(
            f"[local] no artifacts found on volume "
            f"(prefix={prefix!r})"
        )
        return
    _save_results_locally(artifacts)


@app.local_entrypoint()
def list_artifacts(prefix: str = ""):
    """List ``(path, size)`` for every file currently on the artifact
    Volume — quick sanity check that a detached run is persisting data.
    """
    with modal.enable_output():
        rows = list_artifacts_in_volume.remote(prefix=prefix)
    if not rows:
        print(f"[local] volume empty (prefix={prefix!r})")
        return
    total = sum(sz for _, sz in rows)
    for rel, sz in rows:
        print(f"  {sz:>12,d}  {rel}")
    print(f"[local] {len(rows)} file(s), {total:,} bytes total")


@app.local_entrypoint()
def build_dataset(
    mode: str = "musique_wikimqa",
    n_per_set: int = 750,
    seed: int = 42,
    out: str = "realistic_qa/inputs/extended_cacheblend.json",
    out_samsum: str = "realistic_qa/inputs/extended_samsum.json",
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    st_device: str = "cuda",
    embed_batch_size: int = 128,
    chat_model: str = "gpt-4.1-mini",
    gateway: str = "",
    top_k: int = 6,
    shuffle_seed: int = 34,
    samsum_n: int = 1500,
    splitter_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
):
    """Run ``realistic_qa/scripts/build_extended_dataset.py`` on a GPU worker (downloads land in image HF cache).

    Uses secrets from the ``compcache-dotenv`` Modal secret (see module docstring).

    Examples::

        modal run modal_runner.py::build_dataset
        modal run modal_runner.py::build_dataset --mode both --n-per-set 50
    """
    with modal.enable_output():
        artifacts = run_build_extended_dataset.remote(
            mode=mode,
            n_per_set=n_per_set,
            seed=seed,
            out=out,
            out_samsum=out_samsum,
            st_model=st_model,
            st_device=st_device,
            embed_batch_size=embed_batch_size,
            chat_model=chat_model,
            gateway=gateway,
            top_k=top_k,
            shuffle_seed=shuffle_seed,
            samsum_n=samsum_n,
            splitter_model=splitter_model,
        )
    _save_results_locally(artifacts)
