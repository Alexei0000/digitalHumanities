"""
llm_client.py
Async HTTP client for a remote Ollama server.

Two-layer retry strategy:
  Layer 1 — network retries: exponential back-off on connection errors (MAX_RETRIES).
  Layer 2 — JSON repair retries: if the response arrives but is not valid JSON,
            send a lightweight "please fix this JSON" prompt before giving up
            (JSON_REPAIR_RETRIES).  This catches small models that add preamble
            text, forget a closing bracket, or wrap the output in markdown.
"""

import asyncio
import logging
import aiohttp

from config import (
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    MAX_CONCURRENT_REQUESTS,
    JSON_REPAIR_RETRIES,
)
from json_utils import extract_json

logger = logging.getLogger(__name__)

_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

_REPAIR_PROMPT = """\
The following text should be valid JSON but is malformed or incomplete.
Return ONLY the corrected, complete JSON. No explanation. No markdown. No extra text.

MALFORMED:
{bad_json}
"""


class OllamaClient:
    """Async client for the Ollama /api/generate endpoint."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OllamaClient":
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()

    # ─── Public API ──────────────────────────────────────────────────────────

    async def generate(self, model: str, prompt: str) -> str:
        """
        Send a prompt and return the raw response string.
        Raises RuntimeError after all retries are exhausted.
        """
        return await self._call(model, prompt)

    async def generate_json(self, model: str, prompt: str) -> dict:
        """
        Send a prompt, parse the response as JSON, and return the dict.
        On parse failure, attempts up to JSON_REPAIR_RETRIES repair calls
        before raising ValueError.
        """
        raw = await self._call(model, prompt)

        # Try to parse immediately
        try:
            return extract_json(raw)
        except ValueError:
            pass

        # JSON repair loop
        bad = raw
        for attempt in range(JSON_REPAIR_RETRIES):
            logger.warning(
                "Bad JSON from %s (repair attempt %d/%d). "
                "First 120 chars: %r",
                model, attempt + 1, JSON_REPAIR_RETRIES, bad[:120],
            )
            repair_prompt = _REPAIR_PROMPT.format(bad_json=bad[:3000])
            bad = await self._call(model, repair_prompt)
            try:
                return extract_json(bad)
            except ValueError:
                continue

        raise ValueError(
            f"Model {model} returned invalid JSON after "
            f"{JSON_REPAIR_RETRIES} repair attempts. "
            f"Response: {bad[:200]!r}"
        )

    # ─── Model validation ────────────────────────────────────────────────────

    async def check_models(self) -> bool:
        """
        Ping Ollama, list available models, and warn about any models in config
        that are not found.  Returns False if Ollama is unreachable.
        """
        from config import (
            CHARACTER_MODEL, DIALOGUE_MODEL, SPEAKER_MODEL,
            LISTENER_MODEL, SENTIMENT_MODEL, ACTIVE_PROFILE,
        )
        required = {
            CHARACTER_MODEL, DIALOGUE_MODEL,
            SPEAKER_MODEL, LISTENER_MODEL, SENTIMENT_MODEL,
        }

        try:
            async with self._session.get(f"{OLLAMA_URL}/api/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.error(
                "Cannot reach Ollama at %s — %s\n"
                "Check that Ollama is running and OLLAMA_URL is correct.",
                OLLAMA_URL, exc,
            )
            return False

        available = {m["name"] for m in data.get("models", [])}
        logger.info(
            "Ollama reachable. Profile: '%s'. Available models: %s",
            ACTIVE_PROFILE,
            ", ".join(sorted(available)) or "(none)",
        )

        missing = required - available
        if missing:
            logger.error(
                "The following models are configured but NOT found on Ollama:\n"
                "  Missing : %s\n"
                "  Available: %s\n"
                "Update ACTIVE_PROFILE in config.py or pull the model with `ollama pull <name>`.",
                ", ".join(sorted(missing)),
                ", ".join(sorted(available)),
            )
            return False

        logger.info("All required models confirmed: %s", ", ".join(sorted(required)))
        return True

    # ─── Internal ────────────────────────────────────────────────────────────

    async def _call(self, model: str, prompt: str) -> str:
        """Raw HTTP call with network-level retry."""
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
                "temperature": 0.1,
                "num_predict": 8192,
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
                        response_text = result.get("response", "")
                        if not response_text.strip():
                            raise RuntimeError(
                                f"Model '{model}' returned an empty response. "
                                f"Check the model name with `ollama list` on your server."
                            )
                        return response_text

                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Ollama attempt %d/%d failed (%s). Retrying in %ds.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"Ollama call failed after {MAX_RETRIES} attempts: {exc}"
                        ) from exc
                    await asyncio.sleep(wait)

        raise RuntimeError("Unexpected exit from retry loop")