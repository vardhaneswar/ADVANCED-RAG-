"""
Build human-readable sentences from market time series data.
"""

from typing import Dict, List


def build_market_sentences(market_data: Dict) -> List[str]:
    """
    Convert market history into simple text sentences.

    Args:
        market_data (Dict): Output from fetch_market_data().

    Returns:
        List[str]: Each string describes one date's prices.
    """
    ticker = market_data.get("ticker", "UNKNOWN")
    history = market_data.get("history", [])

    sentences: List[str] = []
    for row in history:
        sentence = (
            f"On {row['date']}, {ticker} opened at {row['open']}, "
            f"high {row['high']}, low {row['low']}, closed at {row['close']}, "
            f"with volume {row['volume']}."
        )
        sentences.append(sentence)

    return sentences
