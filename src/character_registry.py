"""
character_registry.py
Novel-wide character registry that accumulates character names and aliases
across all chunks, then resolves duplicates before export.

Merging strategy (in order):
1. Exact name match.
2. Title-stripped match  (Mr. Darcy → Darcy).
3. Alias cross-match     (Lizzy is an alias of Elizabeth → merge).
"""

import re
import uuid
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(
    r"^(Mr\.|Mrs\.|Ms\.|Miss|Dr\.|Sir|Lady|Lord|Captain|Col\.|Prof\.)\s+",
    re.IGNORECASE,
)


def _strip_titles(name: str) -> str:
    return _TITLE_RE.sub("", name).strip()


class CharacterRegistry:
    """
    Accumulates characters extracted across chunks for one novel.
    Call add_character() repeatedly; call export() at the end.
    """

    def __init__(self) -> None:
        # canonical_name → set of aliases
        self._registry: dict[str, set[str]] = defaultdict(set)

    # ─── Ingestion ───────────────────────────────────────────────────────────

    def add_character(self, name: str, aliases: list[str]) -> None:
        """
        Add a character and merge with any existing entry that matches
        by name or alias.
        """
        name = name.strip()
        aliases = {a.strip() for a in aliases if a.strip()}

        canonical = self._find_canonical(name, aliases)

        if canonical is None:
            # New character
            self._registry[name] = aliases
        else:
            # Merge into existing entry
            self._registry[canonical].update(aliases)
            self._registry[canonical].add(name)

    def _find_canonical(
        self, name: str, aliases: set[str]
    ) -> str | None:
        """
        Return the existing canonical name that matches `name` or any alias,
        or None if this is genuinely a new character.
        """
        all_names = {name} | aliases
        stripped = {_strip_titles(n) for n in all_names}

        for canonical, existing_aliases in self._registry.items():
            existing_all = {canonical} | existing_aliases
            existing_stripped = {_strip_titles(n) for n in existing_all}

            # Direct or title-stripped overlap
            if all_names & existing_all or stripped & existing_stripped:
                return canonical

        return None

    # ─── Export ──────────────────────────────────────────────────────────────

    def export(self) -> list[dict]:
        """
        Returns a list of character dicts ready for characters.json:
            [{id, name, aliases}, ...]
        """
        result = []
        for name, aliases in sorted(self._registry.items()):
            result.append(
                {
                    "id": f"c_{uuid.uuid5(uuid.NAMESPACE_DNS, name).hex[:8]}",
                    "name": name,
                    "aliases": sorted(aliases),
                }
            )
        return result

    def get_all_names(self) -> set[str]:
        """Return every known name (canonical + aliases) for prompt injection."""
        names: set[str] = set()
        for canonical, aliases in self._registry.items():
            names.add(canonical)
            names.update(aliases)
        return names

    def resolve(self, name: str) -> str:
        """
        Resolve an alias back to its canonical name.
        Returns the input unchanged if not found.
        """
        name = name.strip()
        if name in self._registry:
            return name
        for canonical, aliases in self._registry.items():
            if name in aliases:
                return canonical
        return name

    def __len__(self) -> int:
        return len(self._registry)