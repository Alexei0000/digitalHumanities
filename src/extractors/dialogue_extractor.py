"""
extractors/dialogue_extractor.py
Calls the LLM to extract all spoken utterances from a single chunk.
"""

import logging
from schemas import DialogueResponse
from json_utils import extract_json
from prompts.dialogue_prompt import DIALOGUE_PROMPT
from config import DIALOGUE_MODEL

logger = logging.getLogger(__name__)


class DialogueExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(self, chunk_text: str) -> DialogueResponse:
        prompt = DIALOGUE_PROMPT.format(text=chunk_text)
        raw = await self.client.generate(DIALOGUE_MODEL, prompt)
        try:
            data = extract_json(raw)
            return DialogueResponse(**data)
        except Exception as exc:
            logger.warning("DialogueExtractor parse error: %s", exc)
            return DialogueResponse(dialogues=[])