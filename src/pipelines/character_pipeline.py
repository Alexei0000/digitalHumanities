"""
pipelines/character_pipeline.py
Runs character extraction over all chunks of one novel
and returns a populated CharacterRegistry.
"""

import logging
from extractors.character_extractor import CharacterExtractor
from character_registry import CharacterRegistry

logger = logging.getLogger(__name__)


class CharacterPipeline:
    def __init__(self, client) -> None:
        self.extractor = CharacterExtractor(client)

    async def run(self, chunks: list[dict]) -> CharacterRegistry:
        registry = CharacterRegistry()

        for chunk in chunks:
            result = await self.extractor.extract(chunk["text"])
            for character in result.characters:
                registry.add_character(character.name, character.aliases)
                logger.debug("Registered: %s", character.name)

        logger.info(
            "CharacterPipeline: %d canonical characters found.", len(registry)
        )
        return registry