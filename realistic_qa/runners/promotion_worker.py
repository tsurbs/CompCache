"""Async background worker that materializes joint pair KV caches.

Given a pair (d_a, d_b) and their tokenized forms, the worker runs a single
collection forward on the concatenation ``tokens_a ‖ tokens_b`` on the same
vLLM engine used by the main eval loop, extracts the per-layer KV slice, and
writes it to the pair store.

A shared :class:`threading.Lock` coordinates GPU access with the main loop:
main-loop forwards and worker forwards never interleave, but the main loop
releases the lock between queries so the worker gets to run. That gives the
triggering query amortized zero cost for promotion while guaranteeing vLLM
never sees concurrent requests.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

from pair_kv_store import PairKVStore


class PromotionJob:
    __slots__ = ("doc_a", "doc_b", "tokens_a", "tokens_b")

    def __init__(
        self, doc_a: str, doc_b: str, tokens_a: List[int], tokens_b: List[int]
    ) -> None:
        self.doc_a = doc_a
        self.doc_b = doc_b
        self.tokens_a = list(tokens_a)
        self.tokens_b = list(tokens_b)


class PromotionWorker(threading.Thread):
    """Runs one pair-concat forward pass per job, writes result to pair store.

    The caller injects ``run_collection_forward`` — a callback that already
    knows how to invoke vLLM with ``collect=True`` and return the per-layer
    KV list for a token sequence. Keeping vLLM outside this module makes the
    worker unit-testable with a stub.
    """

    def __init__(
        self,
        pair_store: PairKVStore,
        run_collection_forward: Callable[[List[int]], list],
        gpu_lock: threading.Lock,
        *,
        max_queue_size: int = 1024,
        log_prefix: str = "[promotion]",
    ) -> None:
        super().__init__(daemon=True)
        self.pair_store = pair_store
        self._run = run_collection_forward
        self.gpu_lock = gpu_lock
        self.queue: "queue.Queue[Optional[PromotionJob]]" = queue.Queue(maxsize=max_queue_size)
        # Must not use ``_stop``: ``threading.Thread`` reserves that name for ``join()``.
        self._stop_event = threading.Event()
        self._log_prefix = log_prefix
        self.completed = 0
        self.errors = 0
        self.dropped = 0

    # ---- public API -------------------------------------------------------

    def enqueue(self, job: PromotionJob) -> bool:
        """Non-blocking enqueue. Drops the job if the queue is full."""
        try:
            self.queue.put_nowait(job)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def stop(self, *, drain: bool = False) -> None:
        """Signal shutdown. If ``drain``, wait for pending jobs to complete."""
        if drain:
            self.queue.join()
        self._stop_event.set()
        # Poison pill to unblock the get() call.
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def promote_sync(self, job: PromotionJob) -> None:
        """Run one job on the calling thread (still under the GPU lock)."""
        self._process(job)

    def stats(self) -> dict:
        return {
            "completed": self.completed,
            "errors": self.errors,
            "dropped": self.dropped,
            "pending": self.queue.qsize(),
            "running": self.is_alive(),
        }

    # ---- thread body ------------------------------------------------------

    def run(self) -> None:  # pragma: no cover - exercised via integration
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            try:
                self._process(item)
            finally:
                self.queue.task_done()

    def _process(self, job: PromotionJob) -> None:
        concat = job.tokens_a + job.tokens_b
        try:
            with self.gpu_lock:
                joint_layers = self._run(concat)
            self.pair_store.put(job.doc_a, job.doc_b, joint_layers)
            self.completed += 1
        except Exception as exc:  # pragma: no cover - logged at runtime
            self.errors += 1
            print(f"{self._log_prefix} FAILED ({job.doc_a}, {job.doc_b}): {exc}", flush=True)
