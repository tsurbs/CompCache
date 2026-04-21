from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runners"))

from pair_kv_store import FullJointPairStore, SparseDeltaPairStore  # noqa: E402
from promotion_worker import PromotionJob, PromotionWorker  # noqa: E402


def _stub_forward(concat_tokens):
    """Pretend to run the model: returns per-layer [K, V] shaped by token length."""
    n = len(concat_tokens)
    return [
        [
            torch.full((n, 2, 4), float(n), dtype=torch.float32),
            torch.full((n, 2, 4), float(-n), dtype=torch.float32),
        ]
        for _ in range(3)
    ]


def test_promote_sync_writes_to_store():
    store = FullJointPairStore(max_entries=4)
    lock = threading.Lock()
    worker = PromotionWorker(store, _stub_forward, lock)

    worker.promote_sync(PromotionJob("d_a", "d_b", [1, 2, 3], [4, 5]))
    assert store.contains("d_a", "d_b")
    got = store.get("d_a", "d_b")
    assert got is not None
    # 3 layers, each tensor has seq_len = 5 (3 + 2).
    assert len(got) == 3
    assert got[0][0].shape == (5, 2, 4)
    assert worker.completed == 1
    assert worker.errors == 0


def test_background_thread_processes_queue():
    store = FullJointPairStore(max_entries=4)
    lock = threading.Lock()
    worker = PromotionWorker(store, _stub_forward, lock)
    worker.start()

    ok = worker.enqueue(PromotionJob("a", "b", [1], [2]))
    assert ok
    worker.enqueue(PromotionJob("c", "d", [3, 4], [5]))
    # Give the worker a chance; it polls with a 250ms timeout.
    deadline = time.time() + 5.0
    while worker.completed < 2 and time.time() < deadline:
        time.sleep(0.05)

    worker.stop(drain=True)
    worker.join(timeout=2.0)
    assert worker.completed == 2
    assert store.contains("a", "b")
    assert store.contains("c", "d")


def test_gpu_lock_is_acquired_during_forward():
    """Main loop holds the lock → worker must wait before running the forward."""
    store = FullJointPairStore(max_entries=4)
    lock = threading.Lock()

    observed_lock_state: list[bool] = []

    def probing_forward(tokens):
        observed_lock_state.append(lock.locked())
        return _stub_forward(tokens)

    worker = PromotionWorker(store, probing_forward, lock)

    # Simulate main loop holding the lock; worker should block inside _process.
    lock.acquire()
    t = threading.Thread(
        target=worker.promote_sync,
        args=(PromotionJob("a", "b", [1], [2]),),
    )
    t.start()
    time.sleep(0.1)
    assert observed_lock_state == []  # forward has NOT run yet
    lock.release()
    t.join(timeout=2.0)
    assert observed_lock_state == [True]  # lock was held during the forward
    assert store.contains("a", "b")


def test_promote_sync_passes_individuals_to_delta_store():
    """PromotionWorker must forward individuals to a needs_individuals store."""
    store = SparseDeltaPairStore(max_entries=2, top_k_ratio=1.0)
    lock = threading.Lock()
    tokens_a = [1, 2, 3]
    tokens_b = [4, 5]
    la, lb = len(tokens_a), len(tokens_b)

    def stub_forward(concat_tokens):
        # Joint = cat(individual_a, individual_b) + small perturbation;
        # here we just return sentinel tensors with the right shape.
        n = len(concat_tokens)
        return [
            [
                torch.full((n, 2, 4), 7.0, dtype=torch.float32),
                torch.full((n, 2, 4), -7.0, dtype=torch.float32),
            ]
            for _ in range(3)
        ]

    individual_a = [
        [torch.full((la, 2, 4), 1.0), torch.full((la, 2, 4), -1.0)]
        for _ in range(3)
    ]
    individual_b = [
        [torch.full((lb, 2, 4), 2.0), torch.full((lb, 2, 4), -2.0)]
        for _ in range(3)
    ]

    worker = PromotionWorker(store, stub_forward, lock)
    job = PromotionJob(
        "d_a",
        "d_b",
        tokens_a,
        tokens_b,
        individual_a=individual_a,
        individual_b=individual_b,
    )
    worker.promote_sync(job)
    assert store.contains("d_a", "d_b")
    # Round-trip check using the same individuals.
    got = store.get("d_a", "d_b", individual_a=individual_a, individual_b=individual_b)
    assert got is not None
    # At ratio=1.0 we expect exact reconstruction → all sevens.
    for k, v in got:
        assert torch.all(k == 7.0)
        assert torch.all(v == -7.0)


def test_failed_forward_increments_errors():
    store = FullJointPairStore(max_entries=2)
    lock = threading.Lock()

    def failing(tokens):
        raise RuntimeError("simulated OOM")

    worker = PromotionWorker(store, failing, lock)
    worker.promote_sync(PromotionJob("a", "b", [1], [2]))
    assert worker.errors == 1
    assert worker.completed == 0
    assert not store.contains("a", "b")
