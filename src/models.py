"""
models.py
Internal data-transfer objects for the pipeline stages.
These are NOT LLM response schemas (those live in schemas.py).
"""

from pydantic import BaseModel, Field
from typing import Optional


class Novel(BaseModel):
    novel_id: str
    path: str
    title: Optional[str] = None


class Scene(BaseModel):
    scene_id: str
    novel_id: str
    scene_index: int
    text: str


class Chunk(BaseModel):
    chunk_id: str
    novel_id: str
    scene_id: str
    chunk_index: int
    text: str


class InteractionRecord(BaseModel):
    """One speaker→listener interaction, ready for graph construction."""
    interaction_id: str
    novel_id: str
    chunk_id: str
    scene_id: str

    speaker: str
    listener: str
    quote: str
    quote_type: str           # "direct" | "indirect"

    sentiment_score: int      # -5 … +5
    emotion: str

    speaker_confidence: float
    listener_confidence: float