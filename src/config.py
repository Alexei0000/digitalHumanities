"""
config.py
All pipeline configuration in one place.
Adjust OLLAMA_URL and model names to match your remote server.

Paths are resolved relative to the PROJECT ROOT (the folder that contains
src/, data/, output/) regardless of which directory you run python from.
"""

from pathlib import Path

# Project root = one level above this file (src/config.py → project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Ollama Remote Server ────────────────────────────────────────────────────
OLLAMA_URL = "http://140.112.180.223:11434"

# ─── Model Selection ────────────────────────────────────────────────────────
# Use smaller models for speed, larger for accuracy.
CHARACTER_MODEL  = "gemma4:31b"
DIALOGUE_MODEL   = "gemma4:31b"
SPEAKER_MODEL    = "gemma4:31b"
LISTENER_MODEL   = "gemma4:31b"
SENTIMENT_MODEL  = "gemma4:31b"

# ─── Concurrency ────────────────────────────────────────────────────────────
# Lower this if Ollama returns 503 / VRAM errors.
MAX_CONCURRENT_REQUESTS = 4

# ─── HTTP Settings ───────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 300   # seconds per LLM call
MAX_RETRIES     = 3     # exponential back-off retries

# ─── Chunking ────────────────────────────────────────────────────────────────
MAX_CHARS_PER_CHUNK     = 8_000
OVERLAP_CHARS           = 1_000

# ─── Scene Segmentation ──────────────────────────────────────────────────────
# Paragraphs per "scene" when no semantic model is used.
PARAGRAPHS_PER_SCENE    = 30

# Cosine-similarity threshold; drop below → scene boundary.
SCENE_SIMILARITY_THRESHOLD = 0.65

# ─── Graph Weights ───────────────────────────────────────────────────────────
# W = DIALOGUE_WEIGHT * dialogue_count
#   + COOCCUR_WEIGHT  * co_occurrence_count
#   + SENTIMENT_WEIGHT * abs(mean_sentiment)
DIALOGUE_WEIGHT  = 1.0
COOCCUR_WEIGHT   = 0.3
SENTIMENT_WEIGHT = 0.5

# ─── Paths ───────────────────────────────────────────────────────────────────
# All paths are absolute so the pipeline works regardless of cwd.
DB_PATH     = str(_PROJECT_ROOT / "pipeline.db")
CORPUS_ROOT = str(_PROJECT_ROOT / "data")
OUTPUT_ROOT = str(_PROJECT_ROOT / "output")