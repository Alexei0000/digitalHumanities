"""
chunker.py
Splits scene text into overlapping chunks suitable for LLM context windows.

Key design choices:
- Breaks only at paragraph boundaries (never mid-sentence).
- Adds a configurable character overlap between consecutive chunks
  so dialogues spanning a boundary aren't lost.
- Returns fully-typed Chunk dicts ready for database insertion.
"""

import uuid
import logging
from config import MAX_CHARS_PER_CHUNK, OVERLAP_CHARS

logger = logging.getLogger(__name__)


class ChunkBuilder:
    """
    Args:
        max_chars:     Maximum characters per chunk (default from config).
        overlap_chars: Characters of overlap between consecutive chunks.
    """

    def __init__(
        self,
        max_chars: int = MAX_CHARS_PER_CHUNK,
        overlap_chars: int = OVERLAP_CHARS,
    ) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def build_chunks(
        self,
        novel_id: str,
        scene_id: str,
        scene_text: str,
    ) -> list[dict]:
        """
        Split scene_text into overlapping chunks.

        Returns:
            List of dicts:
                {chunk_id, novel_id, scene_id, chunk_index, text}
        """
        paragraphs = [
            p.strip()
            for p in scene_text.split("\n\n")
            if p.strip()
        ]

        chunks = []
        current_paras: list[str] = []
        current_len = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para) + 2  # +2 for \n\n separator

            # Flush if adding this paragraph would exceed the limit
            if current_paras and current_len + para_len > self.max_chars:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(
                    self._make_chunk(
                        novel_id, scene_id, chunk_index, chunk_text
                    )
                )
                chunk_index += 1

                # Overlap: keep the tail of the current window
                overlap_text = chunk_text[-self.overlap_chars :]
                current_paras = [overlap_text]
                current_len = len(overlap_text)

            current_paras.append(para)
            current_len += para_len

        # Final chunk
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(
                self._make_chunk(novel_id, scene_id, chunk_index, chunk_text)
            )

        logger.debug(
            "scene %s → %d chunks", scene_id, len(chunks)
        )
        return chunks

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_chunk(
        novel_id: str,
        scene_id: str,
        chunk_index: int,
        text: str,
    ) -> dict:
        return {
            "chunk_id": str(uuid.uuid4()),
            "novel_id": novel_id,
            "scene_id": scene_id,
            "chunk_index": chunk_index,
            "text": text,
        }