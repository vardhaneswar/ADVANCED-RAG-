"""
Embeddings Adapter (dummy)
--------------------------
Later this will call a real embedding model.
For now it just makes length-based fake vectors.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingResult:
    text: str
    vector: list[float]


def dummy_embed_texts(texts: List[str]) -> List[EmbeddingResult]:
    """
    Dummy embedding: vector = [len, len/2, len/3].

    Args:
        texts (List[str]): Text chunks.

    Returns:
        List[EmbeddingResult]
    """
    results: List[EmbeddingResult] = []

    for t in texts:
        length = float(len(t))
        vec = [length, length / 2.0, length / 3.0]
        results.append(EmbeddingResult(text=t, vector=vec))

    return results
