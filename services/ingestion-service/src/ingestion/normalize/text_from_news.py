"""
Build human-readable sentences from news items.
"""

from typing import List, Dict


def build_news_sentences(news_items: List[Dict]) -> List[str]:
    """
    Convert news items into short text snippets.

    Args:
        news_items (List[Dict]): Output from fetch_news_for_ticker().

    Returns:
        List[str]: One sentence per news item.
    """
    sentences: List[str] = []

    for item in news_items:
        ticker = item.get("ticker", "UNKNOWN")
        title = item.get("title", "")
        publisher = item.get("publisher", "")
        sentence = f"News about {ticker} from {publisher}: {title}"
        sentences.append(sentence)

    return sentences
