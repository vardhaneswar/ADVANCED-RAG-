"""
Context Fusion
--------------
Combine vector and graph retrieval results into a single
ordered context list.
"""

from typing import List, Dict
from .vector_retriever import RetrievedDoc


def fuse_results(
    vector_docs: List[RetrievedDoc],
    graph_facts: List[Dict],
) -> Dict:
    """
    Merge vector and graph results into a single context structure.

    Args:
        vector_docs (List[RetrievedDoc]): docs from vector index
        graph_facts (List[Dict]): relationships from graph DB

    Returns:
        Dict: {
            "documents": [...],
            "graph": [...]
        }
    """
    print("[fusion] Fusing vector and graph results...")

    docs_payload = [
        {
            "text": d.text,
            "score": d.score,
            "source": d.source,
        }
        for d in vector_docs
    ]

    fusion_result = {
        "documents": docs_payload,
        "graph": graph_facts,
    }

    print(f"[fusion] {len(docs_payload)} docs, {len(graph_facts)} graph facts in fused context.")
    return fusion_result
