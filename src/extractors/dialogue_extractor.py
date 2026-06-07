import logging
from schemas import DialogueResponse
from prompts.dialogue_prompt import DIALOGUE_PROMPT
from config import DIALOGUE_MODEL

logger = logging.getLogger(__name__)


class DialogueExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(self, chunk_text: str) -> DialogueResponse:
        prompt = DIALOGUE_PROMPT.format(text=chunk_text)
        try:
            data = await self.client.generate_json(DIALOGUE_MODEL, prompt)
            return DialogueResponse(**data)
        except Exception as exc:
            logger.warning("DialogueExtractor failed: %s", exc)
            return DialogueResponse(dialogues=[])