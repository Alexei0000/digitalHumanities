"""
pipelines/interaction_pipeline.py
For each extracted dialogue, sequentially runs:
    1. Speaker attribution
    2. Listener attribution
    3. Sentiment analysis

Produces the interaction_log — the source of truth for graph construction.

Design notes:
- Skips interactions where speaker or listener is UNKNOWN and both
  confidences are below MIN_CONFIDENCE.
- Resolves speaker/listener names through the CharacterRegistry so that
  "Lizzy" becomes "Elizabeth Bennet" in the final output.
- Saves every interaction to the database for crash-resumability.
"""

import asyncio
import logging
import uuid

from extractors.speaker_extractor import SpeakerExtractor
from extractors.listener_extractor import ListenerExtractor
from extractors.sentiment_extractor import SentimentExtractor
from character_registry import CharacterRegistry

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.4    # discard interactions below this threshold


class InteractionPipeline:
    def __init__(self, client) -> None:
        self.speaker_ext  = SpeakerExtractor(client)
        self.listener_ext = ListenerExtractor(client)
        self.sentiment_ext = SentimentExtractor(client)

    async def run(
        self,
        dialogues: list[dict],
        registry: CharacterRegistry,
        db=None,           # optional Database instance for persistence
    ) -> list[dict]:
        """
        Args:
            dialogues: Output from DialoguePipeline.run()
            registry:  Populated CharacterRegistry for this novel.
            db:        Database instance (pass to enable per-interaction saves).

        Returns:
            List of interaction dicts ready for interaction_log.json and CSV export.
        """
        character_names = registry.get_all_names()
        interactions = []

        for idx, dialogue in enumerate(dialogues):
            quote      = dialogue["quote"]
            chunk_text = dialogue.get("chunk_text", "")
            novel_id   = dialogue["novel_id"]
            chunk_id   = dialogue["chunk_id"]
            scene_id   = dialogue["scene_id"]

            logger.debug(
                "[%d/%d] Attributing: %r", idx + 1, len(dialogues), quote[:60]
            )

            # ── Speaker ──────────────────────────────────────────────────
            speaker_resp = await self.speaker_ext.extract(
                quote, chunk_text, character_names
            )
            speaker = registry.resolve(speaker_resp.speaker)

            # ── Listener ─────────────────────────────────────────────────
            listener_resp = await self.listener_ext.extract(
                quote, speaker, chunk_text, character_names
            )
            listener = registry.resolve(listener_resp.listener)

            # ── Filter low-confidence unknowns ───────────────────────────
            if (
                speaker == "UNKNOWN"
                and listener == "UNKNOWN"
                and speaker_resp.confidence < MIN_CONFIDENCE
                and listener_resp.confidence < MIN_CONFIDENCE
            ):
                logger.debug("Skipping low-confidence interaction.")
                continue

            # ── Sentiment ────────────────────────────────────────────────
            sentiment_resp = await self.sentiment_ext.extract(
                quote, speaker, listener, chunk_text
            )

            interaction = {
                "interaction_id":      str(uuid.uuid4()),
                "novel_id":            novel_id,
                "chunk_id":            chunk_id,
                "scene_id":            scene_id,
                "speaker":             speaker,
                "listener":            listener,
                "quote":               quote,
                "quote_type":          dialogue["quote_type"],
                "sentiment_score":     sentiment_resp.score,
                "emotion":             sentiment_resp.emotion,
                "speaker_confidence":  round(speaker_resp.confidence, 4),
                "listener_confidence": round(listener_resp.confidence, 4),
            }

            interactions.append(interaction)

            # ── Persist to DB if available ────────────────────────────────
            if db is not None:
                db.save_interaction(**interaction)

        logger.info(
            "InteractionPipeline: %d interactions recorded.", len(interactions)
        )
        return interactions