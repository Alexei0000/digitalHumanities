"""
schemas.py
Pydantic schemas that validate every LLM JSON response.
If the model returns garbage, validation raises an error before
it can corrupt the database.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Character Extraction ────────────────────────────────────────────────────

class Character(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)


class CharacterResponse(BaseModel):
    characters: list[Character]


# ─── Dialogue Extraction ─────────────────────────────────────────────────────

class Dialogue(BaseModel):
    quote: str
    quote_type: str                        # "direct" | "indirect"
    start_char: Optional[int] = None       # character offset in chunk text
    end_char: Optional[int] = None


class DialogueResponse(BaseModel):
    dialogues: list[Dialogue]


# ─── Speaker Attribution ─────────────────────────────────────────────────────

class SpeakerResponse(BaseModel):
    speaker: str                           # canonical name or "UNKNOWN"
    confidence: float = Field(ge=0.0, le=1.0)


# ─── Listener Attribution ────────────────────────────────────────────────────

class ListenerResponse(BaseModel):
    listener: str                          # canonical name or "UNKNOWN"
    confidence: float = Field(ge=0.0, le=1.0)


# ─── Sentiment Extraction ────────────────────────────────────────────────────

class SentimentResponse(BaseModel):
    score: int = Field(ge=-5, le=5)        # -5 hostile … +5 strongly positive
    emotion: str                           # e.g. "hostility", "affection"