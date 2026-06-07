"""
llm_client.py
Async HTTP client for a remote Ollama server.

Two-layer retry strategy:
  Layer 1 — network retries: exponential back-off on connection/HTTP errors.
  Layer 2 — JSON repair: if response arrives but isn't valid JSON, re-prompt
            with a repair request before giving up.

Key design decisions:
  - NO "format":"json" — causes empty responses on qwen3.5 and some other models.
    We use prompt instructions + extract_json() instead.
  - Repair prompt uses % substitution, NOT .format(), to avoid { } collisions
    with JSON content in the bad response.
  - Empty response is treated as a retryable error, not a fatal one.
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

# NOTE: uses %s substitution, NOT .format(), so JSON braces in bad_json
# don't get misinterpreted as format placeholders.
_REPAIR_TEMPLATE = (
    "The text below should be a valid JSON object but is malformed or has extra text.\n"
    "Output ONLY the corrected JSON object. No explanation. No markdown. No extra text.\n\n"
    "MALFORMED INPUT:\n"
    "%s"
)


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
        """Send a prompt and return the raw response string."""
        return await self._call(model, prompt)

    async def generate_json(self, model: str, prompt: str) -> dict:
        """
        Send a prompt, return parsed JSON dict.
        On parse failure, retries with a repair prompt up to JSON_REPAIR_RETRIES times.
        Raises ValueError if all attempts fail.
        """
        raw = await self._call(model, prompt)

        try:
            return extract_json(raw)
        except ValueError:
            pass

        # JSON repair loop
        bad = raw
        for attempt in range(JSON_REPAIR_RETRIES):
            logger.warning(
                "Bad JSON from %s (repair attempt %d/%d). First 120 chars: %r",
                model, attempt + 1, JSON_REPAIR_RETRIES, bad[:120],
            )
            repair_prompt = _REPAIR_TEMPLATE % bad[:3000]
            bad = await self._call(model, repair_prompt)
            try:
                return extract_json(bad)
            except ValueError:
                continue

        raise ValueError(
            "Model %s returned invalid JSON after %d repair attempts. "
            "Response: %r" % (model, JSON_REPAIR_RETRIES, bad[:200])
        )

    # ─── Model Validation ────────────────────────────────────────────────────

    async def check_models(self) -> bool:
        """
        Ping Ollama and verify all configured models exist.
        Returns False (and logs errors) if anything is wrong.
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
                "Check OLLAMA_URL in config.py.",
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
                "Models configured but NOT found on Ollama:\n"
                "  Missing:   %s\n"
                "  Available: %s\n"
                "Fix ACTIVE_PROFILE in config.py or run `ollama pull <name>`.",
                ", ".join(sorted(missing)),
                ", ".join(sorted(available)),
            )
            return False

        logger.info("All required models confirmed: %s", ", ".join(sorted(required)))
        return True

    # ─── Internal ────────────────────────────────────────────────────────────

    async def _call(self, model: str, prompt: str) -> str:
        """
        Raw HTTP call with network-level retry.
        Empty responses are retried, not silently accepted.
        """
        if self._session is None:
            raise RuntimeError(
                "OllamaClient must be used as an async context manager."
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # No "format":"json" — causes empty output on qwen3.5 and similar models.
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
                        text = result.get("response", "").strip()
                        if not text:
                            raise RuntimeError(
                                "Empty response from model '%s'. "
                                "The model may be overloaded or the prompt too long." % model
                            )
                        return text

                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "Ollama attempt %d/%d failed (%s). Retrying in %ds.",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(
                            "Ollama call failed after %d attempts: %s" % (MAX_RETRIES, exc)
                        ) from exc
                    await asyncio.sleep(wait)

        raise RuntimeError("Unexpected exit from retry loop")