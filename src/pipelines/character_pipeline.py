"""
pipelines/character_pipeline.py
Runs character extraction over all chunks concurrently (bounded by semaphore).
"""

import asyncio
import logging
from extractors.character_extractor import CharacterExtractor
from character_registry import CharacterRegistry
from config import MAX_CONCURRENT_REQUESTS

logger = logging.getLogger(__name__)


class CharacterPipeline:
    def __init__(self, client) -> None:
        self.extractor = CharacterExtractor(client)

    async def run(self, chunks: list[dict]) -> CharacterRegistry:
        registry = CharacterRegistry()
        sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def process_chunk(chunk):
            async with sem:
                return await self.extractor.extract(chunk["text"])

        results = await asyncio.gather(
            *[process_chunk(c) for c in chunks],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.warning("CharacterPipeline chunk error: %s", result)
                continue
            for character in result.characters:
                registry.add_character(character.name, character.aliases)

        logger.info("CharacterPipeline: %d canonical characters found.", len(registry))
        return registry