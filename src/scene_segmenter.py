"""
scene_segmenter.py
Splits a novel's raw text into scenes.

Strategy (two-tier):
1. Explicit delimiters: ***, ---, chapter headers, Part I, etc.
2. Paragraph grouping fallback: group N paragraphs per scene.
   (Cosine-similarity segmentation is plugged in as an optional upgrade
    via SceneSegmenter.use_embeddings=True, but requires sentence-transformers.)
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Explicit scene boundary patterns (case-insensitive, full line)
EXPLICIT_BREAK_PATTERNS = [
    r"^\*{3,}$",
    r"^-{3,}$",
    r"^_{3,}$",
    r"^#{1,3}\s+",                          # Markdown headings
    r"^Chapter\s+\d+",
    r"^Chapter\s+[IVXLC]+",
    r"^CHAPTER\s+",
    r"^Part\s+\d+",
    r"^Part\s+[IVXLC]+",
    r"^Book\s+\d+",
    r"^Epilogue",
    r"^Prologue",
    r"^Scene\s+\d+",
]

_BREAK_RE = re.compile(
    "|".join(f"(?:{p})" for p in EXPLICIT_BREAK_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


class SceneSegmenter:
    """
    Converts a raw novel string into a list of scene strings.

    Args:
        paragraphs_per_scene: Fallback grouping size when no explicit
                              delimiters are found.
        use_embeddings:       Enable cosine-similarity scene detection.
                              Requires sentence-transformers installed.
        similarity_threshold: Scene boundary if sim(p_i, p_i+1) < threshold.
    """

    def __init__(
        self,
        paragraphs_per_scene: int = 30,
        use_embeddings: bool = False,
        similarity_threshold: float = 0.65,
    ) -> None:
        self.paragraphs_per_scene = paragraphs_per_scene
        self.use_embeddings = use_embeddings
        self.similarity_threshold = similarity_threshold
        self._embedder = None

        if use_embeddings:
            self._load_embedder()

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("Embedding model loaded: BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to paragraph-count segmentation."
            )
            self.use_embeddings = False

    # ─── Public API ──────────────────────────────────────────────────────────

    def segment(self, text: str) -> list[str]:
        """
        Split text into scenes. Returns a list of scene strings.
        """
        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return []

        # Try explicit delimiters first
        scenes = self._split_by_explicit_markers(paragraphs)
        if len(scenes) > 1:
            logger.debug("Explicit delimiter segmentation: %d scenes", len(scenes))
            return scenes

        # Embedding-based segmentation
        if self.use_embeddings and self._embedder and len(paragraphs) >= 4:
            scenes = self._split_by_embeddings(paragraphs)
            logger.debug("Embedding segmentation: %d scenes", len(scenes))
            return scenes

        # Fallback: fixed paragraph groups
        scenes = self._split_by_count(paragraphs)
        logger.debug("Paragraph-count segmentation: %d scenes", len(scenes))
        return scenes

    # ─── Internal Methods ────────────────────────────────────────────────────

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines; strip and filter empty paragraphs."""
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _split_by_explicit_markers(self, paragraphs: list[str]) -> list[str]:
        """Group paragraphs by explicit scene-break lines."""
        scenes: list[list[str]] = [[]]
        for para in paragraphs:
            if _BREAK_RE.match(para):
                if scenes[-1]:          # don't open empty scenes
                    scenes.append([])
            else:
                scenes[-1].append(para)
        return ["\n\n".join(s) for s in scenes if s]

    def _split_by_embeddings(self, paragraphs: list[str]) -> list[str]:
        """
        Embed each paragraph; insert a scene boundary where
        cosine similarity between adjacent paragraphs drops below threshold.
        """
        import numpy as np

        embeddings = self._embedder.encode(
            paragraphs, batch_size=64, show_progress_bar=False
        )

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        scenes: list[list[str]] = [[paragraphs[0]]]
        for i in range(1, len(paragraphs)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if sim < self.similarity_threshold:
                scenes.append([])
            scenes[-1].append(paragraphs[i])

        return ["\n\n".join(s) for s in scenes if s]

    def _split_by_count(self, paragraphs: list[str]) -> list[str]:
        """Simple N-paragraph groups."""
        scenes = []
        for i in range(0, len(paragraphs), self.paragraphs_per_scene):
            chunk = paragraphs[i : i + self.paragraphs_per_scene]
            scenes.append("\n\n".join(chunk))
        return scenes