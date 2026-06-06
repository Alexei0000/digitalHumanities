"""
llm_client.py
Async HTTP client for a remote Ollama server.

Features:
- Async context manager (use with `async with OllamaClient() as client`)
- Enforces JSON-only responses via Ollama's `format: "json"` option
- Exponential back-off on transient errors
- Global semaphore limits concurrent in-flight requests
"""

import asyncio
import logging
import aiohttp

from config import (
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    MAX_CONCURRENT_REQUESTS,
)

logger = logging.getLogger(__name__)

# One semaphore shared across the entire process.
_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class OllamaClient:
    """Async client for Ollama /api/generate endpoint."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OllamaClient":
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    async def generate(self, model: str, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the raw response string.

        Args:
            model:  Ollama model name, e.g. "qwen3:32b"
            prompt: Full prompt text.

        Returns:
            Raw response string from the model (may need JSON extraction).

        Raises:
            RuntimeError: after MAX_RETRIES failed attempts.
        """
        if self._session is None:
            raise RuntimeError(
                "OllamaClient must be used as an async context manager."
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,   # low temp → more consistent JSON
                "num_predict": 8192,  # max tokens in response; prevents truncation
            },
        }

        async with _SEMAPHORE:
            for attempt in range(MAX_RETRIES):
                try:
                    async with self._session.post(
                        f"{OLLAMA_URL}/api/generate",
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
                        return result["response"]

                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Ollama attempt %d/%d failed (%s). Retrying in %ds.",
                        attempt + 1,
                        MAX_RETRIES,
                        exc,
                        wait,
                    )
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"Ollama call failed after {MAX_RETRIES} attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(wait)

        # unreachable, but satisfies type checkers
        raise RuntimeError("Unexpected exit from retry loop")