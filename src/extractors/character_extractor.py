"""
extractors/character_extractor.py
Calls the LLM to extract characters from a single chunk.
"""

import logging
from schemas import CharacterResponse
from json_utils import extract_json
from prompts.character_prompt import CHARACTER_PROMPT
from config import CHARACTER_MODEL

logger = logging.getLogger(__name__)


class CharacterExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(self, chunk_text: str) -> CharacterResponse:
        prompt = CHARACTER_PROMPT.format(text=chunk_text)
        raw = await self.client.generate(CHARACTER_MODEL, prompt)
        try:
            data = extract_json(raw)
            return CharacterResponse(**data)
        except Exception as exc:
            logger.warning("CharacterExtractor parse error: %s", exc)
            return CharacterResponse(characters=[])