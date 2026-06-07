import logging
from schemas import SentimentResponse
from prompts.sentiment_prompt import SENTIMENT_PROMPT
from config import SENTIMENT_MODEL

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 400


class SentimentExtractor:
    def __init__(self, client) -> None:
        self.client = client

    async def extract(self, quote: str, speaker: str, listener: str, chunk_text: str) -> SentimentResponse:
        context = self._extract_context(quote, chunk_text)
        prompt = SENTIMENT_PROMPT.format(
            speaker=speaker, listener=listener, quote=quote, context=context
        )
        try:
            data = await self.client.generate_json(SENTIMENT_MODEL, prompt)
            return SentimentResponse(**data)
        except Exception as exc:
            logger.warning("SentimentExtractor failed: %s", exc)
            return SentimentResponse(score=0, emotion="neutral")

    @staticmethod
    def _extract_context(quote: str, chunk_text: str) -> str:
        idx = chunk_text.find(quote)
        if idx == -1:
            return chunk_text[:_CONTEXT_WINDOW]
        start = max(0, idx - _CONTEXT_WINDOW)
        end = min(len(chunk_text), idx + len(quote) + _CONTEXT_WINDOW)
        return chunk_text[start:end]