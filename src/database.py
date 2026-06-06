"""
database.py
SQLite persistence layer.

Tables:
    novels            – discovered .txt/.md files
    chunks            – chunked text segments
    characters        – extracted character entities
    dialogues         – extracted speech acts
    interactions      – speaker→listener events with sentiment
    processing_state  – per-novel, per-stage completion flags (resume support)

All writes use INSERT OR IGNORE / INSERT OR REPLACE so the pipeline
is idempotent: re-running a stage won't duplicate rows.
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = "pipeline.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")   # better concurrency

    # ─── Schema ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create all tables if they don't exist."""
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS novels (
                novel_id TEXT PRIMARY KEY,
                path     TEXT NOT NULL,
                title    TEXT,
                status   TEXT DEFAULT 'pending'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                novel_id    TEXT NOT NULL,
                scene_id    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS characters (
                character_id TEXT PRIMARY KEY,
                novel_id     TEXT NOT NULL,
                name         TEXT NOT NULL,
                aliases      TEXT          -- JSON array string
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dialogues (
                dialogue_id TEXT PRIMARY KEY,
                novel_id    TEXT NOT NULL,
                chunk_id    TEXT NOT NULL,
                quote       TEXT NOT NULL,
                quote_type  TEXT NOT NULL,
                start_char  INTEGER,
                end_char    INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id      TEXT PRIMARY KEY,
                novel_id            TEXT NOT NULL,
                chunk_id            TEXT NOT NULL,
                scene_id            TEXT NOT NULL,
                speaker             TEXT NOT NULL,
                listener            TEXT NOT NULL,
                quote               TEXT NOT NULL,
                quote_type          TEXT NOT NULL,
                sentiment_score     INTEGER NOT NULL,
                emotion             TEXT NOT NULL,
                speaker_confidence  REAL NOT NULL,
                listener_confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS processing_state (
                novel_id  TEXT NOT NULL,
                stage     TEXT NOT NULL,
                completed INTEGER DEFAULT 0,
                PRIMARY KEY (novel_id, stage)
            )
            """,
        ]
        for stmt in stmts:
            self.conn.execute(stmt)
        self.conn.commit()
        logger.info("Database initialized.")

    # ─── Novels ──────────────────────────────────────────────────────────────

    def add_novel(self, novel_id: str, path: str, title: str = "") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO novels VALUES (?, ?, ?, 'pending')",
            (novel_id, path, title),
        )
        self.conn.commit()

    def get_pending_novels(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM novels WHERE status = 'pending'"
        ).fetchall()

    def mark_novel_done(self, novel_id: str) -> None:
        self.conn.execute(
            "UPDATE novels SET status = 'done' WHERE novel_id = ?",
            (novel_id,),
        )
        self.conn.commit()

    # ─── Chunks ──────────────────────────────────────────────────────────────

    def save_chunk(
        self,
        chunk_id: str,
        novel_id: str,
        scene_id: str,
        chunk_index: int,
        text: str,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?)",
            (chunk_id, novel_id, scene_id, chunk_index, text),
        )
        self.conn.commit()

    def get_chunks_for_novel(self, novel_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE novel_id = ? ORDER BY chunk_index",
            (novel_id,),
        ).fetchall()

    # ─── Characters ──────────────────────────────────────────────────────────

    def save_character(
        self,
        character_id: str,
        novel_id: str,
        name: str,
        aliases: list[str],
    ) -> None:
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO characters VALUES (?, ?, ?, ?)",
            (character_id, novel_id, name, json.dumps(aliases, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_characters_for_novel(self, novel_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM characters WHERE novel_id = ?",
            (novel_id,),
        ).fetchall()

    # ─── Dialogues ───────────────────────────────────────────────────────────

    def save_dialogue(
        self,
        dialogue_id: str,
        novel_id: str,
        chunk_id: str,
        quote: str,
        quote_type: str,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO dialogues VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dialogue_id, novel_id, chunk_id, quote, quote_type, start_char, end_char),
        )
        self.conn.commit()

    def get_dialogues_for_novel(self, novel_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM dialogues WHERE novel_id = ?",
            (novel_id,),
        ).fetchall()

    # ─── Interactions ────────────────────────────────────────────────────────

    def save_interaction(
        self,
        interaction_id: str,
        novel_id: str,
        chunk_id: str,
        scene_id: str,
        speaker: str,
        listener: str,
        quote: str,
        quote_type: str,
        sentiment_score: int,
        emotion: str,
        speaker_confidence: float,
        listener_confidence: float,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO interactions VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                interaction_id, novel_id, chunk_id, scene_id,
                speaker, listener, quote, quote_type,
                sentiment_score, emotion,
                speaker_confidence, listener_confidence,
            ),
        )
        self.conn.commit()

    def get_interactions_for_novel(self, novel_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM interactions WHERE novel_id = ?",
            (novel_id,),
        ).fetchall()

    # ─── Processing State (Resume Support) ───────────────────────────────────

    def is_stage_complete(self, novel_id: str, stage: str) -> bool:
        row = self.conn.execute(
            "SELECT completed FROM processing_state WHERE novel_id=? AND stage=?",
            (novel_id, stage),
        ).fetchone()
        return bool(row and row["completed"])

    def mark_stage_complete(self, novel_id: str, stage: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO processing_state
               VALUES (?, ?, 1)""",
            (novel_id, stage),
        )
        self.conn.commit()