from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

from pair_kv_store import PairKVStore, StackedLayers

class PromotionJob:
    __slots__ = (
        "doc_a",
        "doc_b",
        "tokens_a",
        "tokens_b",
        "individual_a",
        "individual_b",
    )

    def __init__(
        self,
        doc_a: str,
        doc_b: str,
        tokens_a: List[int],
        tokens_b: List[int],
        *,
        individual_a: Optional[StackedLayers] = None,
        individual_b: Optional[StackedLayers] = None,
    ) -> None:
        self.doc_a = doc_a
        self.doc_b = doc_b
        self.tokens_a = list(tokens_a)
        self.tokens_b = list(tokens_b)
        
        
        self.individual_a = individual_a
        self.individual_b = individual_b

class PromotionWorker(threading.Thread):

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
        
        self._stop_event = threading.Event()
        self._log_prefix = log_prefix
        self.completed = 0
        self.errors = 0
        self.dropped = 0

    

    def enqueue(self, job: PromotionJob) -> bool:
        try:
            self.queue.put_nowait(job)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def stop(self, *, drain: bool = False) -> None:
        if drain:
            self.queue.join()
        self._stop_event.set()
        
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

    def promote_sync(self, job: PromotionJob) -> None:
        self._process(job)

    def stats(self) -> dict:
        return {
            "completed": self.completed,
            "errors": self.errors,
            "dropped": self.dropped,
            "pending": self.queue.qsize(),
            "running": self.is_alive(),
        }

    

    def run(self) -> None:  
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
        
        
        
        
        
        
        
        try:
            if self.pair_store.contains(job.doc_a, job.doc_b):
                self.completed += 1
                return
        except Exception:
            
            
            pass

        concat = job.tokens_a + job.tokens_b
        try:
            with self.gpu_lock:
                joint_layers = self._run(concat)
            self.pair_store.put(
                job.doc_a,
                job.doc_b,
                joint_layers,
                individual_a=job.individual_a,
                individual_b=job.individual_b,
            )
            self.completed += 1
        except Exception as exc:  
            self.errors += 1
            print(f"{self._log_prefix} FAILED ({job.doc_a}, {job.doc_b}): {exc}", flush=True)
        finally:
            
            
            
            
            
            
            try:
                del joint_layers  # noqa: F821 — may not exist if pre-check fired
            except Exception:
                pass
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
