"""
main.py
Entry point for the Novel SNA pipeline.

Processes every .txt/.md file under CORPUS_ROOT and writes per-novel outputs:
    output/<novel_id>/
        characters.json
        dialogues.json
        interaction_log.json
        nodes.csv
        edges.csv

Resume support: if a novel's stage is already marked complete in the database,
that stage is skipped. Delete pipeline.db to start fresh.

Usage:
    cd src/
    python main.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# ─── Ensure src/ is on the path when running from project root ───────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import CORPUS_ROOT, OUTPUT_ROOT, DB_PATH
from database import Database
from corpus_loader import CorpusLoader
from scene_segmenter import SceneSegmenter
from chunker import ChunkBuilder
from llm_client import OllamaClient
from character_registry import CharacterRegistry

from pipelines.character_pipeline import CharacterPipeline
from pipelines.dialogue_pipeline import DialoguePipeline
from pipelines.interaction_pipeline import InteractionPipeline

from graph.graph_builder import GraphBuilder
from exporters.json_exporter import JSONExporter
from exporters.csv_exporter import CSVExporter

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─── Pipeline ────────────────────────────────────────────────────────────────

async def process_novel(
    novel: dict,
    db: Database,
    client: OllamaClient,
) -> None:
    novel_id = novel["novel_id"]
    path     = novel["path"]
    title    = novel.get("title", novel_id)

    output_dir = Path(OUTPUT_ROOT) / novel_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("━━━ [%s] %s ━━━", novel_id[:8], title)

    # ── Read text ─────────────────────────────────────────────────────────
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return

    # ── Segment + chunk (always re-run; cheap) ────────────────────────────
    segmenter = SceneSegmenter()
    chunker   = ChunkBuilder()
    chunks: list[dict] = []

    scenes = segmenter.segment(text)
    for s_idx, scene_text in enumerate(scenes):
        scene_id = f"{novel_id}_s{s_idx:04d}"
        for chunk in chunker.build_chunks(novel_id, scene_id, scene_text):
            db.save_chunk(
                chunk["chunk_id"],
                chunk["novel_id"],
                chunk["scene_id"],
                chunk["chunk_index"],
                chunk["text"],
            )
            chunks.append(chunk)

    logger.info("Segmented into %d scenes → %d chunks.", len(scenes), len(chunks))

    # ── Stage 1: Character Extraction ─────────────────────────────────────
    if not db.is_stage_complete(novel_id, "characters"):
        logger.info("Stage 1: Character extraction …")
        char_pipeline = CharacterPipeline(client)
        registry: CharacterRegistry = await char_pipeline.run(chunks)

        for char in registry.export():
            db.save_character(
                char["id"], novel_id, char["name"], char["aliases"]
            )

        JSONExporter.save(registry.export(), output_dir / "characters.json")
        db.mark_stage_complete(novel_id, "characters")
        logger.info("Stage 1 done: %d characters.", len(registry))
    else:
        logger.info("Stage 1 already done (resuming).")
        # Rebuild registry from DB
        registry = CharacterRegistry()
        import json as _json
        for row in db.get_characters_for_novel(novel_id):
            registry.add_character(
                row["name"],
                _json.loads(row["aliases"] or "[]"),
            )

    # ── Stage 2: Dialogue Extraction ──────────────────────────────────────
    if not db.is_stage_complete(novel_id, "dialogues"):
        logger.info("Stage 2: Dialogue extraction …")
        dial_pipeline = DialoguePipeline(client)
        dialogues: list[dict] = await dial_pipeline.run(chunks)

        for d in dialogues:
            db.save_dialogue(
                d["dialogue_id"],
                d["novel_id"],
                d["chunk_id"],
                d["quote"],
                d["quote_type"],
            )

        # Save without chunk_text (internal field)
        export_dialogues = [
            {k: v for k, v in d.items() if k != "chunk_text"}
            for d in dialogues
        ]
        JSONExporter.save(export_dialogues, output_dir / "dialogues.json")
        db.mark_stage_complete(novel_id, "dialogues")
        logger.info("Stage 2 done: %d dialogues.", len(dialogues))
    else:
        logger.info("Stage 2 already done (resuming).")
        dialogues = []
        for row in db.get_dialogues_for_novel(novel_id):
            # Recover chunk_text from chunks table for attribution context
            chunk_rows = db.get_chunks_for_novel(novel_id)
            chunk_map  = {r["chunk_id"]: r["text"] for r in chunk_rows}
            dialogues.append(
                {
                    "dialogue_id": row["dialogue_id"],
                    "novel_id":    row["novel_id"],
                    "chunk_id":    row["chunk_id"],
                    "scene_id":    "unknown",    # scene_id not stored in dialogues table
                    "quote":       row["quote"],
                    "quote_type":  row["quote_type"],
                    "chunk_text":  chunk_map.get(row["chunk_id"], ""),
                }
            )

    # ── Stage 3: Speaker / Listener / Sentiment ───────────────────────────
    if not db.is_stage_complete(novel_id, "interactions"):
        logger.info("Stage 3: Interaction attribution …")
        int_pipeline = InteractionPipeline(client)
        interactions: list[dict] = await int_pipeline.run(dialogues, registry, db=db)

        JSONExporter.save(interactions, output_dir / "interaction_log.json")
        db.mark_stage_complete(novel_id, "interactions")
        logger.info("Stage 3 done: %d interactions.", len(interactions))
    else:
        logger.info("Stage 3 already done (resuming).")
        interactions = [dict(r) for r in db.get_interactions_for_novel(novel_id)]

    # ── Stage 4: Graph Construction ───────────────────────────────────────
    logger.info("Stage 4: Building graph …")
    import json as _json
    characters_export = []
    for row in db.get_characters_for_novel(novel_id):
        characters_export.append(
            {
                "id":      row["character_id"],
                "name":    row["name"],
                "aliases": _json.loads(row["aliases"] or "[]"),
            }
        )

    builder = GraphBuilder(characters_export, interactions)
    nodes_df, edges_df = builder.build()

    CSVExporter.save(nodes_df, output_dir / "nodes.csv")
    CSVExporter.save(edges_df, output_dir / "edges.csv")
    db.mark_novel_done(novel_id)

    logger.info(
        "✓ Novel complete: %d nodes, %d edges → %s",
        len(nodes_df),
        len(edges_df),
        output_dir,
    )


async def main(reset_ids: set[str], reset_stage: str) -> None:
    db = Database(DB_PATH)
    db.initialize()

    # ── Apply resets before processing ───────────────────────────────────────
    if reset_ids:
        stages = (
            [reset_stage] if reset_stage != "all"
            else ["characters", "dialogues", "interactions"]
        )
        for novel_id in reset_ids:
            for stage in stages:
                db.conn.execute(
                    "DELETE FROM processing_state WHERE novel_id=? AND stage=?",
                    (novel_id, stage),
                )
            # Also reset novel status so it's re-processed
            db.conn.execute(
                "UPDATE novels SET status='pending' WHERE novel_id=?",
                (novel_id,),
            )
        db.conn.commit()
        logger.info(
            "Reset stage '%s' for %d novel(s): %s",
            reset_stage,
            len(reset_ids),
            ", ".join(s[:8] for s in reset_ids),
        )

    loader = CorpusLoader(CORPUS_ROOT)
    novels = loader.discover()

    if not novels:
        logger.warning("No .txt/.md files found under '%s'.", CORPUS_ROOT)
        return

    # Register all discovered novels
    for novel in novels:
        db.add_novel(novel["novel_id"], novel["path"], novel.get("title", ""))

    logger.info("Corpus: %d novels discovered.", len(novels))

    async with OllamaClient() as client:
        # ── Startup: verify Ollama is reachable and models exist ─────────────
        if not await client.check_models():
            return   # errors already logged inside check_models()

        for novel in novels:
            try:
                await process_novel(novel, db, client)
            except Exception as exc:
                logger.error(
                    "Failed on novel %s: %s", novel["novel_id"][:8], exc,
                    exc_info=True,
                )
                # Continue to next novel rather than crashing the whole run


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Novel SNA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal run (resumes automatically)
  python main.py

  # Reset ALL stages for one novel and reprocess it
  python main.py --reset f33b44f6f7cc99c30d13b5d1e6bba1b0

  # Reset only the dialogue stage for one novel
  python main.py --reset f33b44f6f7cc99c30d13b5d1e6bba1b0 --stage dialogues

  # Reset multiple novels
  python main.py --reset f33b44f6f7cc99c30d13b5d1e6bba1b0 2484413e...

  # Reset ALL novels (full reprocess — same as deleting pipeline.db)
  python main.py --reset all
        """,
    )
    parser.add_argument(
        "--reset",
        nargs="+",
        metavar="NOVEL_ID",
        default=[],
        help=(
            "Novel ID(s) to reset (first 8 chars or full hash). "
            "Use 'all' to reset every novel."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["all", "characters", "dialogues", "interactions"],
        default="all",
        help="Which stage to reset (default: all stages).",
    )

    args = parser.parse_args()

    # Resolve short IDs and 'all' keyword
    reset_ids: set[str] = set()
    if args.reset:
        if "all" in args.reset:
            # Will be resolved after DB init; pass sentinel
            reset_ids = {"__ALL__"}
        else:
            reset_ids = set(args.reset)

    async def _main_wrapper(raw_ids: set, stage: str) -> None:
        resolved = raw_ids

        if "__ALL__" in raw_ids:
            db_temp = Database(DB_PATH)
            db_temp.initialize()
            rows = db_temp.conn.execute("SELECT novel_id FROM novels").fetchall()
            resolved = {r["novel_id"] for r in rows}
            logger.info("--reset all: will reset %d novels.", len(resolved))

        elif raw_ids:
            db_temp = Database(DB_PATH)
            db_temp.initialize()
            all_ids = {
                r["novel_id"]
                for r in db_temp.conn.execute("SELECT novel_id FROM novels").fetchall()
            }
            expanded = set()
            for rid in raw_ids:
                if rid in all_ids:
                    expanded.add(rid)
                else:
                    matches = {full for full in all_ids if full.startswith(rid)}
                    if matches:
                        expanded.update(matches)
                    else:
                        logger.warning("No novel found matching ID prefix: %s", rid)
            resolved = expanded

        await main(resolved, stage)

    asyncio.run(_main_wrapper(reset_ids, args.stage))