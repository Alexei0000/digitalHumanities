"""
pipelines/_queue.py
True bounded worker pool — at most N tasks ever exist at once.

Unlike asyncio.gather (which creates all coroutines immediately), this uses
asyncio.Queue so only N chunks are in-flight at any time. When Ollama starts
timing out, the adaptive throttle detects the failure rate and automatically
reduces concurrency and adds delays between calls.
"""

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class _AdaptiveThrottle:
    """
    Tracks recent failures and slows down when Ollama is struggling.

    Failure rate is measured over a sliding window of recent calls.
    If failure rate exceeds THRESHOLD, concurrency is halved and a delay
    is added between calls. Recovers gradually as calls succeed.
    """
    WINDOW       = 20     # look at last N calls
    THRESHOLD    = 0.4    # if >40% fail, throttle
    MIN_DELAY    = 0.0    # seconds between calls (normal)
    MAX_DELAY    = 5.0    # seconds between calls (overloaded)
    MIN_WORKERS  = 1
    RECOVERY_STEP = 0.5   # reduce delay by this much per successful call

    def __init__(self, base_concurrency: int) -> None:
        self.base_concurrency = base_concurrency
        self._history: deque = deque(maxlen=self.WINDOW)
        self.delay = self.MIN_DELAY
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self._history.append(True)
            self.delay = max(self.MIN_DELAY, self.delay - self.RECOVERY_STEP)

    async def record_failure(self) -> None:
        async with self._lock:
            self._history.append(False)
            failure_rate = self._history.count(False) / len(self._history)
            if failure_rate >= self.THRESHOLD:
                self.delay = min(self.MAX_DELAY, self.delay + 1.0)
                logger.warning(
                    "Throttle: %.0f%% failure rate over last %d calls. "
                    "Adding %.1fs delay between requests.",
                    failure_rate * 100, len(self._history), self.delay,
                )

    async def wait(self) -> None:
        if self.delay > 0:
            await asyncio.sleep(self.delay)


async def _run_queue(
    extract_fn,
    chunks: list[dict],
    concurrency: int,
) -> list:
    """
    Process chunks using a true worker pool.
    At most `concurrency` workers are active at any time.
    Workers pull from a shared queue; no more than concurrency
    connections are ever open simultaneously.

    Returns results in the same order as input chunks.
    Failed chunks return None.
    """
    if not chunks:
        return []

    total = len(chunks)
    results = [None] * total
    throttle = _AdaptiveThrottle(concurrency)

    # Use asyncio.Queue as the work queue
    queue: asyncio.Queue = asyncio.Queue()
    for i, chunk in enumerate(chunks):
        await queue.put((i, chunk))

    completed = 0
    start_time = time.time()

    async def worker() -> None:
        nonlocal completed
        while True:
            try:
                idx, chunk = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            await throttle.wait()
            try:
                results[idx] = await extract_fn(chunk["text"])
                await throttle.record_success()
            except Exception as exc:
                await throttle.record_failure()
                logger.warning("Chunk %d/%d failed: %s", idx + 1, total, exc)
            finally:
                completed += 1
                queue.task_done()

                # Progress log every 50 chunks
                if completed % 50 == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    logger.info(
                        "Progress: %d/%d chunks (%.1f/s, ~%.0fs remaining, "
                        "throttle delay: %.1fs)",
                        completed, total, rate, remaining, throttle.delay,
                    )

    # Start exactly `concurrency` workers
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)
    return results