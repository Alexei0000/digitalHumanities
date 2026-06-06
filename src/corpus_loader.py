"""
corpus_loader.py
Recursively discovers .txt and .md files under a root directory.
Generates a stable MD5-based novel_id from the file path so that
re-runs produce identical IDs.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_novel_id(path: Path) -> str:
    """Stable ID derived from the absolute path."""
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()


def infer_title(path: Path) -> str:
    """Best-effort title from filename (strip extension, replace underscores)."""
    return path.stem.replace("_", " ").replace("-", " ").title()


class CorpusLoader:
    """
    Walks a directory tree and yields (novel_id, path, title) tuples
    for every .txt and .md file found.
    """

    EXTENSIONS = (".txt", ".md")

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def discover(self) -> list[dict]:
        """
        Returns a list of dicts:
            {novel_id, path, title}
        Sorted by path for deterministic ordering.
        """
        found = []

        for path in sorted(self.root_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS:
                novel_id = generate_novel_id(path)
                found.append(
                    {
                        "novel_id": novel_id,
                        "path": str(path),
                        "title": infer_title(path),
                    }
                )

        logger.info(
            "CorpusLoader: discovered %d novels under %s",
            len(found),
            self.root_dir,
        )
        return found