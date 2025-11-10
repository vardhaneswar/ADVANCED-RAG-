"""
News API Client (using yfinance.news)
-------------------------------------
Simple way to grab latest news for a ticker.
"""

from typing import List, Dict
import yfinance as yf


def fetch_news_for_ticker(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Fetch recent news items for a ticker.

    Args:
        ticker (str): e.g. "NVDA", "AAPL".
        limit (int): Max number of items.

    Returns:
        List[Dict]: Each item has at least: title, publisher, link, providerPublishTime.
    """
    print(f"[news_api_client] Fetching news for {ticker} (limit={limit})...")
    ticker_obj = yf.Ticker(ticker)

    try:
        news_items = ticker_obj.news or []
    except Exception as e:
        print(f"[news_api_client] Error fetching news for {ticker}: {e}")
        return []

    trimmed: List[Dict] = []
    for item in news_items[:limit]:
        trimmed.append(
            {
                "ticker": ticker,
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "providerPublishTime": item.get("providerPublishTime", 0),
            }
        )

    print(f"[news_api_client] Retrieved {len(trimmed)} news items for {ticker}.")
    return trimmed
