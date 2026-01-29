from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from functools import cached_property
from typing import TYPE_CHECKING, Self, Callable, Any

if TYPE_CHECKING:
    pass


class Submit:
    """
    Manages parallel file I/O with double-buffering and two-tier thread pools.

    Architecture:
    - Coordinator pool: Small pool (2 threads) that waits for CUDA events
      and dispatches I/O work. These threads block on event.synchronize()
      but don't do heavy I/O themselves.
    - I/O pool: Larger pool that performs actual file writes in parallel.

    This separation prevents deadlock from nested submissions while allowing
    maximum parallelism for file I/O.

    Double-buffering scheme:
    - `_prev`: Futures from the previous minibatch (being drained)
    - `_curr`: Futures from the current minibatch (being accumulated)

    Call `rotate()` between minibatches to wait for previous work to complete
    and swap buffers. This ensures we stay at most one minibatch ahead of disk I/O,
    preventing unbounded memory growth from queued write operations.
    """

    def __init__(self, workers: int = 8):
        self.workers = workers
        self._coord_pool: ThreadPoolExecutor | None = None
        self._io_pool: ThreadPoolExecutor | None = None
        self._prev: list[Future] = []
        self._curr: list[Future] = []
        self._errors: list[BaseException] = []

    @cached_property
    def coord_pool(self) -> ThreadPoolExecutor:
        """Pool for coordinating I/O tasks (waiting on CUDA events, etc)."""
        return ThreadPoolExecutor(max_workers=4)

    @cached_property
    def io_pool(self) -> ThreadPoolExecutor:
        """Pool for actual file I/O operations."""
        return ThreadPoolExecutor(max_workers=self.workers)

    def __enter__(self) -> Self:
        return self

    def __exit__( self, exc_type, exc, tb) -> None:
        try:
            self._drain(self._prev)
            self._drain(self._curr)

            if self._errors and exc is None:
                raise self._errors[0]

        finally:
            if self._coord_pool is not None:
                self._coord_pool.shutdown(wait=True, cancel_futures=False)
                self._coord_pool = None
            if self._io_pool is not None:
                self._io_pool.shutdown(wait=True, cancel_futures=False)
                self._io_pool = None

    def _drain(self, futures: list[Future]) -> None:
        """Drains a list of futures, collecting any exceptions."""
        if not futures:
            return
        done, _ = wait(futures)
        for f in done:
            try:
                f.result()
            except BaseException as e:
                self._errors.append(e)
        futures.clear()

    def rotate(self) -> None:
        """
        Wait for previous batch to complete, then swap curr -> prev.

        Call this at the start of each minibatch iteration to ensure
        disk I/O doesn't fall more than one batch behind GPU processing.
        """
        self._drain(self._prev)
        self._prev = self._curr
        self._curr = []

    def submit_io(
            self,
            fn: Callable[..., Any],
            *args,
            **kwargs,
    ) -> Future:
        """
        Submit a file I/O task to the I/O pool.

        Use this for actual file write operations. These run in parallel
        across the I/O pool workers.
        """
        return self.io_pool.submit(fn, *args, **kwargs)

    def submit_batch(
            self,
            fn: Callable[..., Any],
            *args,
            **kwargs,
    ) -> None:
        """
        Submit a coordination task that manages a batch of writes.

        The coordination task typically:
        1. Waits for a CUDA event (GPU->CPU transfer complete)
        2. Submits individual file writes via submit_io()
        3. Waits for those writes to complete

        The future is tracked in _curr and will be drained on rotate() or exit.
        """
        future = self.coord_pool.submit(fn, *args, **kwargs)
        self._curr.append(future)
