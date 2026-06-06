"""
extractors/speaker_extractor.py
Calls the LLM to attribute a dialogue quote to a speaker.
Injects surrounding context (±500 chars) and the character registry.
"""

import logging
from schemas import SpeakerResponse
from json_utils import extract_json
from prompts.speaker_prompt import SPEAKER_PROMPT
from config import SPEAKER_MODEL

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 500   # characters of surrounding text


class SpeakerExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(
        self,
        quote: str,
        chunk_text: str,
        character_names: set[str],
    ) -> SpeakerResponse:
        """
        Args:
            quote:           The dialogue utterance to attribute.
            chunk_text:      Full text of the chunk containing the quote.
            character_names: All known names for this novel (registry).
        """
        context = self._extract_context(quote, chunk_text)
        char_list = ", ".join(sorted(character_names)) or "Unknown"

        prompt = SPEAKER_PROMPT.format(
            character_list=char_list,
            quote=quote,
            context=context,
        )
        raw = await self.client.generate(SPEAKER_MODEL, prompt)
        try:
            data = extract_json(raw)
            return SpeakerResponse(**data)
        except Exception as exc:
            logger.warning("SpeakerExtractor parse error: %s", exc)
            return SpeakerResponse(speaker="UNKNOWN", confidence=0.0)

    @staticmethod
    def _extract_context(quote: str, chunk_text: str) -> str:
        idx = chunk_text.find(quote)
        if idx == -1:
            return chunk_text[:_CONTEXT_WINDOW]
        start = max(0, idx - _CONTEXT_WINDOW)
        end = min(len(chunk_text), idx + len(quote) + _CONTEXT_WINDOW)
        return chunk_text[start:end]