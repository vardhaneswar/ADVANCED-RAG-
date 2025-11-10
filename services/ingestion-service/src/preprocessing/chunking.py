"""
Text Chunking
-------------
Simple character-based chunking.
"""

from typing import List


def simple_chunk(text: str, max_chars: int = 500) -> List[str]:
    """
    Split `text` into chunks of at most `max_chars`.

    Args:
        text (str): Cleaned text.
        max_chars (int): Max characters per chunk.

    Returns:
        List[str]: List of chunks.
    """
    chunks: List[str] = []
    current = 0
    length = len(text)

    while current < length:
        chunk = text[current: current + max_chars]
        chunks.append(chunk)
        current += max_chars

    return chunks
