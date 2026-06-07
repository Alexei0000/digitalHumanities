"""
config.py
All pipeline configuration in one place.

QUICK START — change only these two lines:
    OLLAMA_URL   = "http://your-server:11434"
    ACTIVE_PROFILE = "gemma4_31b"   # or "gemma4_12b" / "fast"

Paths are resolved relative to the project root (folder containing src/).
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Ollama Remote Server ─────────────────────────────────────────────────────
OLLAMA_URL = "http://140.112.180.223:11434"

# ─── Model Profiles ───────────────────────────────────────────────────────────
# Each profile sets the model AND adjusts chunk size to match model capability.
# Smaller models need shorter chunks to produce reliable JSON.
#
# Add your own profiles freely.  Keys used downstream:
#   character_model, dialogue_model, speaker_model,
#   listener_model, sentiment_model,
#   max_chars_per_chunk, overlap_chars

MODEL_PROFILES = {

    # Best quality — use for final research runs
    "gemma4_31b": {
        "character_model":   "gemma4:31b",
        "dialogue_model":    "gemma4:31b",
        "speaker_model":     "gemma4:31b",
        "listener_model":    "gemma4:31b",
        "sentiment_model":   "gemma4:31b",
        "max_chars_per_chunk": 8_000,
        "overlap_chars":       1_000,
    },

    # Good balance — character/dialogue with smaller model, attribution with larger
    "gemma4_mixed": {
        "character_model":   "qwen3.5:9b",
        "dialogue_model":    "qwen3.5:9b",
        "speaker_model":     "gemma4:31b",   # attribution needs more reasoning
        "listener_model":    "gemma4:31b",
        "sentiment_model":   "gemma4:31b",
        "max_chars_per_chunk": 5_000,        # smaller chunks for the 12b model
        "overlap_chars":       800,
    },

    # Fast — smaller model, much shorter chunks, good for exploratory runs
    "gemma4_12b": {
        "character_model":   "gemma4:12b",
        "dialogue_model":    "gemma4:12b",
        "speaker_model":     "gemma4:12b",
        "listener_model":    "gemma4:12b",
        "sentiment_model":   "gemma4:12b",
        "max_chars_per_chunk": 3_500,        # short enough for reliable JSON output
        "overlap_chars":       500,
    },

    # Qwen profiles
    "qwen3_32b": {
        "character_model":   "qwen3:32b",
        "dialogue_model":    "qwen3:32b",
        "speaker_model":     "qwen3:32b",
        "listener_model":    "qwen3:32b",
        "sentiment_model":   "qwen3:32b",
        "max_chars_per_chunk": 8_000,
        "overlap_chars":       1_000,
    },

    "qwen3_14b": {
        "character_model":   "qwen3:14b",
        "dialogue_model":    "qwen3:14b",
        "speaker_model":     "qwen3:14b",
        "listener_model":    "qwen3:14b",
        "sentiment_model":   "qwen3:14b",
        "max_chars_per_chunk": 4_000,
        "overlap_chars":       600,
    },
}

# ─── Active Profile ───────────────────────────────────────────────────────────
# Change this one line to switch models for the entire pipeline.
ACTIVE_PROFILE = "gemma4_mixed"

# ─── Resolve profile into flat config variables ───────────────────────────────
_profile = MODEL_PROFILES[ACTIVE_PROFILE]

CHARACTER_MODEL        = _profile["character_model"]
DIALOGUE_MODEL         = _profile["dialogue_model"]
SPEAKER_MODEL          = _profile["speaker_model"]
LISTENER_MODEL         = _profile["listener_model"]
SENTIMENT_MODEL        = _profile["sentiment_model"]
MAX_CHARS_PER_CHUNK    = _profile["max_chars_per_chunk"]
OVERLAP_CHARS          = _profile["overlap_chars"]

# ─── Concurrency ─────────────────────────────────────────────────────────────
# Lower if Ollama returns 503 or VRAM errors.
MAX_CONCURRENT_REQUESTS = 4

# ─── HTTP / Retry ─────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 300   # seconds per LLM call
MAX_RETRIES      = 3     # network-level retries (exponential back-off)
JSON_REPAIR_RETRIES = 2  # extra attempts with simplified repair prompt on bad JSON

# ─── Scene Segmentation ───────────────────────────────────────────────────────
PARAGRAPHS_PER_SCENE       = 30
SCENE_SIMILARITY_THRESHOLD = 0.65

# ─── Graph Weights ────────────────────────────────────────────────────────────
DIALOGUE_WEIGHT  = 1.0
COOCCUR_WEIGHT   = 0.3
SENTIMENT_WEIGHT = 0.5

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_PATH     = str(_PROJECT_ROOT / "pipeline.db")
CORPUS_ROOT = str(_PROJECT_ROOT / "data")
OUTPUT_ROOT = str(_PROJECT_ROOT / "output")