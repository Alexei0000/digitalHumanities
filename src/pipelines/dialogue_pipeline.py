"""
pipelines/dialogue_pipeline.py
Runs dialogue extraction over all chunks of one novel.
Returns a list of dialogue dicts enriched with chunk metadata.
"""

import logging
import uuid
from extractors.dialogue_extractor import DialogueExtractor

logger = logging.getLogger(__name__)


class DialoguePipeline:
    def __init__(self, client) -> None:
        self.extractor = DialogueExtractor(client)

    async def run(self, chunks: list[dict]) -> list[dict]:
        """
        Returns:
            List of dicts:
                {dialogue_id, novel_id, chunk_id, scene_id,
                 quote, quote_type}
        """
        all_dialogues = []

        for chunk in chunks:
            result = await self.extractor.extract(chunk["text"])

            for dialogue in result.dialogues:
                all_dialogues.append(
                    {
                        "dialogue_id": str(uuid.uuid4()),
                        "novel_id": chunk["novel_id"],
                        "chunk_id": chunk["chunk_id"],
                        "scene_id": chunk["scene_id"],
                        "quote": dialogue.quote,
                        "quote_type": dialogue.quote_type,
                        "chunk_text": chunk["text"],   # kept for attribution context
                    }
                )

        if all_dialogues:
            logger.info(
                "DialoguePipeline: %d dialogues extracted.", len(all_dialogues)
            )
        else:
            logger.warning(
                "DialoguePipeline: 0 dialogues extracted from %d chunks. "
                "Check quote style in this novel.",
                len(chunks),
            )
        return all_dialogues