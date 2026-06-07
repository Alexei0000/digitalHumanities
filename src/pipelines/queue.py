"""
pipelines/_queue.py
Bounded async worker queue.
Processes items N at a time; never more than N concurrent Ollama calls.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def _run_queue(extract_fn, chunks: list[dict], concurrency: int) -> list:
    """
    Run extract_fn(chunk["text"]) for every chunk, at most `concurrency` at a time.
    Returns a list of results in the same order as chunks.
    None is inserted for any chunk that raises an exception.
    """
    results = [None] * len(chunks)
    sem = asyncio.Semaphore(concurrency)

    async def worker(idx, chunk):
        async with sem:
            try:
                results[idx] = await extract_fn(chunk["text"])
            except Exception as exc:
                logger.warning("Queue worker %d failed: %s", idx, exc)

    await asyncio.gather(*[worker(i, c) for i, c in enumerate(chunks)])
    return results