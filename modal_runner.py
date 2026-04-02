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

# 128 GiB RAM, 1 TiB ephemeral disk (Modal uses MiB for both request knobs).
_MEM_MIB = 128 * 1024
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

_FN = dict(
    gpu=_GPU,
    memory=_MEM_MIB,
    ephemeral_disk=_DISK_MIB,
    cpu=16.0,
    timeout=86400,
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
    """Run one benchmark with unbuffered Python so logs stream to Modal."""
    print(f"[modal] running {path.name} …", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(path)],
        cwd=_CONTAINER_REPO,
        check=True,
    )


def _collect_result_jsons() -> dict[str, bytes]:
    """Gather all analysis JSON files produced during the run."""
    results = {}
    for inputs_dir, prefix in (
        (Path(_CONTAINER_INPUTS), "standard_qa/inputs/"),
        (Path(_CONTAINER_REALISTIC_INPUTS), "realistic_qa/inputs/"),
    ):
        for pattern in (
            "*_coretrieval.json",
            "*_synthetic_zipf.json",
            "*_ttft_warmup.json",
            "*_ttft_warmup.png",
            "*_ttft_hist.json",
            "*_ttft_hist.png",
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
    os.environ["PYTHONPATH"] = f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}"

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
) -> dict[str, bytes]:
    """FIFO CacheBlend evaluation on a realistic_qa JSON (see ``blend_realistic.py``).

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
        f"[modal] realistic QA: dataset={dataset} fifo_max={fifo_max} "
        f"skip_first={skip_first} max_ctx_len={max_ctx_len} "
        f"max_model_len={max_model_len or '(auto or HF)'} "
        f"gpu_memory_utilization={gpu_memory_utilization}",
        flush=True,
    )
    _invoke_blend_realistic_main()
    return _collect_result_jsons()


@app.function(**_fn_with_workspace_secrets(_FN))
def run_all_blends() -> dict[str, bytes]:
    _maybe_pin_cuda_visible_device0()
    os.environ["PYTHONPATH"] = f"/CompCache/vllm_blend:{_CONTAINER_RUNNERS}"

    ex_dir = Path(_CONTAINER_RUNNERS)
    paths = sorted(ex_dir.glob(BLEND_PATTERN))
    print(f"[modal] run_all_blends: {len(paths)} script(s)", flush=True)
    for path in paths:
        _mistral_only(path)
        _run_blend_script(path)
    return _collect_result_jsons()


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
):
    """Run extended / FIFO CacheBlend on Modal.

    Default is the committed smoke fixture. For the full 6K benchmark, build locally then::

        modal run modal_runner.py::realistic --dataset realistic_qa/inputs/extended_cacheblend.json

    On ~48GB GPUs, keep default ``max_ctx_len`` (8192) or lower; use ``--max-ctx-len 0``
    only if you have enough VRAM for full Mistral-32k-style profiles.

    Examples::

        modal run modal_runner.py::realistic
        modal run modal_runner.py::realistic --skip-first 1000
    """
    with modal.enable_output():
        artifacts = run_realistic_blend.remote(
            dataset=dataset,
            fifo_max=fifo_max,
            skip_first=skip_first,
            max_ctx_len=max_ctx_len,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    _save_results_locally(artifacts)


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
