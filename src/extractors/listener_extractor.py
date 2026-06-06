"""
extractors/listener_extractor.py
Calls the LLM to identify the intended listener of a dialogue quote.
"""

import logging
from schemas import ListenerResponse
from json_utils import extract_json
from prompts.listener_prompt import LISTENER_PROMPT
from config import LISTENER_MODEL

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 500


class ListenerExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(
        self,
        quote: str,
        speaker: str,
        chunk_text: str,
        character_names: set[str],
    ) -> ListenerResponse:
        context = self._extract_context(quote, chunk_text)
        char_list = ", ".join(sorted(character_names)) or "Unknown"

        prompt = LISTENER_PROMPT.format(
            character_list=char_list,
            speaker=speaker,
            quote=quote,
            context=context,
        )
        raw = await self.client.generate(LISTENER_MODEL, prompt)
        try:
            data = extract_json(raw)
            return ListenerResponse(**data)
        except Exception as exc:
            logger.warning("ListenerExtractor parse error: %s", exc)
            return ListenerResponse(listener="UNKNOWN", confidence=0.0)

    @staticmethod
    def _extract_context(quote: str, chunk_text: str) -> str:
        idx = chunk_text.find(quote)
        if idx == -1:
            return chunk_text[:_CONTEXT_WINDOW]
        start = max(0, idx - _CONTEXT_WINDOW)
        end = min(len(chunk_text), idx + len(quote) + _CONTEXT_WINDOW)
        return chunk_text[start:end]