"""
pipelines/character_pipeline.py
Runs character extraction with a bounded worker queue.
MAX_CONCURRENT_REQUESTS workers run in parallel; all other chunks wait.
This prevents blasting Ollama with 380 simultaneous requests.
"""

import asyncio
import logging
from extractors.character_extractor import CharacterExtractor
from character_registry import CharacterRegistry
from config import MAX_CONCURRENT_REQUESTS
from pipelines._queue import _run_queue

logger = logging.getLogger(__name__)


class CharacterPipeline:
    def __init__(self, client) -> None:
        self.extractor = CharacterExtractor(client)

    async def run(self, chunks: list[dict]) -> CharacterRegistry:
        registry = CharacterRegistry()
        results = await _run_queue(self.extractor.extract, chunks, MAX_CONCURRENT_REQUESTS)

        for result in results:
            if result is None:
                continue
            for character in result.characters:
                registry.add_character(character.name, character.aliases)

        logger.info("CharacterPipeline: %d canonical characters found.", len(registry))
        return registry