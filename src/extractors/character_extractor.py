import logging
from schemas import CharacterResponse
from prompts.character_prompt import CHARACTER_PROMPT
from config import CHARACTER_MODEL

logger = logging.getLogger(__name__)


class CharacterExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(self, chunk_text: str) -> CharacterResponse:
        prompt = CHARACTER_PROMPT.format(text=chunk_text)
        try:
            data = await self.client.generate_json(CHARACTER_MODEL, prompt)
            return CharacterResponse(**data)
        except Exception as exc:
            logger.warning("CharacterExtractor failed: %s", exc)
            return CharacterResponse(characters=[])