"""
Vector Indexer (stub)
---------------------
Will later push embeddings into Pinecone or another vector DB.
"""

from typing import List
from .embeddings_adapter import EmbeddingResult


def index_vectors(vectors: List[EmbeddingResult], index_name: str = "finance-demo-index"):
    """
    Placeholder for vector indexing.

    Args:
        vectors (List[EmbeddingResult]): Embeddings to store.
        index_name (str): Index / namespace name.
    """
    print(f"[vector_indexer] Would push {len(vectors)} embeddings to vector DB '{index_name}'.")
    for i, v in enumerate(vectors[:3], start=1):
        print(f"  - Preview {i}: vector={v.vector}")
