"""
pipelines/dialogue_pipeline.py
Runs dialogue extraction with a bounded worker queue.
"""

import asyncio
import logging
import uuid
from extractors.dialogue_extractor import DialogueExtractor
from config import MAX_CONCURRENT_REQUESTS
from pipelines._queue import _run_queue

logger = logging.getLogger(__name__)


class DialoguePipeline:
    def __init__(self, client) -> None:
        self.extractor = DialogueExtractor(client)

    async def run(self, chunks: list[dict]) -> list[dict]:
        results = await _run_queue(self.extractor.extract, chunks, MAX_CONCURRENT_REQUESTS)

        all_dialogues = []
        for chunk, result in zip(chunks, results):
            if result is None:
                continue
            for d in result.dialogues:
                all_dialogues.append({
                    "dialogue_id": str(uuid.uuid4()),
                    "novel_id":    chunk["novel_id"],
                    "chunk_id":    chunk["chunk_id"],
                    "scene_id":    chunk["scene_id"],
                    "quote":       d.quote,
                    "quote_type":  d.quote_type,
                    "chunk_text":  chunk["text"],
                })

        if all_dialogues:
            logger.info("DialoguePipeline: %d dialogues extracted.", len(all_dialogues))
        else:
            logger.warning(
                "DialoguePipeline: 0 dialogues from %d chunks. "
                "Check quote style in this novel.", len(chunks),
            )
        return all_dialogues