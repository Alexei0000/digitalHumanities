"""
pipelines/dialogue_pipeline.py
Runs dialogue extraction over all chunks concurrently (bounded by semaphore).
"""

import asyncio
import logging
import uuid
from extractors.dialogue_extractor import DialogueExtractor
from config import MAX_CONCURRENT_REQUESTS

logger = logging.getLogger(__name__)


class DialoguePipeline:
    def __init__(self, client) -> None:
        self.extractor = DialogueExtractor(client)

    async def run(self, chunks: list[dict]) -> list[dict]:
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def process_chunk(chunk):
            async with sem:
                result = await self.extractor.extract(chunk["text"])
                return [
                    {
                        "dialogue_id": str(uuid.uuid4()),
                        "novel_id":    chunk["novel_id"],
                        "chunk_id":    chunk["chunk_id"],
                        "scene_id":    chunk["scene_id"],
                        "quote":       d.quote,
                        "quote_type":  d.quote_type,
                        "chunk_text":  chunk["text"],
                    }
                    for d in result.dialogues
                ]

        results = await asyncio.gather(
            *[process_chunk(c) for c in chunks],
            return_exceptions=True,
        )

        all_dialogues = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("DialoguePipeline chunk error: %s", result)
                continue
            all_dialogues.extend(result)

        if all_dialogues:
            logger.info("DialoguePipeline: %d dialogues extracted.", len(all_dialogues))
        else:
            logger.warning(
                "DialoguePipeline: 0 dialogues extracted from %d chunks. "
                "Check quote style in this novel.", len(chunks),
            )
        return all_dialogues