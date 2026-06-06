"""
graph/graph_builder.py
Constructs the Social Network Analysis graph from interaction_log data.

Hybrid edge weight formula:
    W = DIALOGUE_WEIGHT  * dialogue_count
      + COOCCUR_WEIGHT   * co_occurrence_count
      + SENTIMENT_WEIGHT * abs(mean_sentiment)

Produces:
    nodes.csv   — CharacterID, Name, Aliases, Degree
    edges.csv   — Source, Target, Interaction_Count, Mean_Sentiment, Weight, Evidence
"""

import logging
from collections import defaultdict

import pandas as pd

from config import DIALOGUE_WEIGHT, COOCCUR_WEIGHT, SENTIMENT_WEIGHT

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, characters: list[dict], interactions: list[dict]) -> None:
        """
        Args:
            characters:   Output of CharacterRegistry.export()
            interactions: Output of InteractionPipeline.run()
        """
        self.characters   = characters
        self.interactions = interactions

    # ─── Public API ──────────────────────────────────────────────────────────

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            (nodes_df, edges_df)
        """
        nodes_df = self._build_nodes()
        edges_df = self._build_edges()

        # Add degree to nodes
        degree = defaultdict(int)
        for _, row in edges_df.iterrows():
            degree[row["Source"]] += 1
            degree[row["Target"]] += 1
        nodes_df["Degree"] = nodes_df["Name"].map(lambda n: degree.get(n, 0))

        return nodes_df, edges_df

    # ─── Internal ────────────────────────────────────────────────────────────

    def _build_nodes(self) -> pd.DataFrame:
        rows = []
        for char in self.characters:
            rows.append(
                {
                    "CharacterID": char["id"],
                    "Name":        char["name"],
                    "Aliases":     ";".join(char.get("aliases", [])),
                }
            )
        df = pd.DataFrame(rows)
        logger.info("Nodes: %d characters.", len(df))
        return df

    def _build_edges(self) -> pd.DataFrame:
        # Aggregate interactions by (speaker, listener) pair
        # Normalise direction: always sort pair so A→B and B→A are separate edges
        # (directed graph; for undirected, sort the pair alphabetically)

        edge_data: dict[tuple, dict] = defaultdict(
            lambda: {
                "interaction_count": 0,
                "sentiment_sum":     0,
                "sentiment_abs_sum": 0,
                "quotes":            [],
            }
        )

        scene_pairs: dict[tuple, set] = defaultdict(set)

        for item in self.interactions:
            speaker  = item["speaker"]
            listener = item["listener"]

            if speaker == "UNKNOWN" or listener == "UNKNOWN":
                continue

            pair = (speaker, listener)
            edge_data[pair]["interaction_count"] += 1
            edge_data[pair]["sentiment_sum"]     += item["sentiment_score"]
            edge_data[pair]["sentiment_abs_sum"] += abs(item["sentiment_score"])
            edge_data[pair]["quotes"].append(item["quote"][:80])

            # Track co-occurrence by scene
            scene_pairs[pair].add(item["scene_id"])

        rows = []
        for (source, target), data in edge_data.items():
            count        = data["interaction_count"]
            mean_sent    = data["sentiment_sum"] / count if count else 0
            cooccur      = len(scene_pairs[(source, target)])

            weight = (
                DIALOGUE_WEIGHT  * count
                + COOCCUR_WEIGHT   * cooccur
                + SENTIMENT_WEIGHT * (data["sentiment_abs_sum"] / count if count else 0)
            )

            rows.append(
                {
                    "Source":            source,
                    "Target":            target,
                    "Interaction_Count": count,
                    "Mean_Sentiment":    round(mean_sent, 3),
                    "Co_Occurrence":     cooccur,
                    "Weight":            round(weight, 3),
                    "Evidence":          " | ".join(data["quotes"][:3]),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Weight", ascending=False)
        logger.info("Edges: %d directed edges.", len(df))
        return df