"""
SEC Filings API Client (stub)
-----------------------------
Placeholder for fetching SEC 10-K / 10-Q filings.

Later you can call EDGAR or another provider.
"""

from typing import List, Dict


def fetch_sec_filings(ticker: str, limit: int = 2) -> List[Dict]:
    """
    Stub: pretend to fetch SEC filings metadata.

    Args:
        ticker (str): Company ticker.
        limit (int): Max number of filings.

    Returns:
        List[Dict]: Fake metadata for now.
    """
    print(f"[sec_api_client] (stub) Would fetch {limit} SEC filings for {ticker}.")
    fake = [
        {
            "ticker": ticker,
            "form": "10-K",
            "year": 2024,
            "url": "https://www.sec.gov/example-10k-url",
        }
    ]
    return fake[:limit]
