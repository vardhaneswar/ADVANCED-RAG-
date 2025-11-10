"""
Graph Indexer (stub)
--------------------
Later will update Neo4j with company relationships.
"""

from typing import List


def update_graph_for_company(ticker: str, related_entities: List[str]):
    """
    Stub: log relationships that would be written to graph DB.

    Args:
        ticker (str): e.g. "NVDA".
        related_entities (List[str]): e.g. ["TSMC", "ASML", "Apple"].
    """
    print(f"[graph_indexer] Would update graph for {ticker}:")
    for entity in related_entities:
        print(f"  - Connects to: {entity}")
