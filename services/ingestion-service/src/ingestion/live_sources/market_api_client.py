"""
Market API Client (REAL using yfinance)
--------------------------------------
Fetches real stock market data using Yahoo Finance (yfinance).
"""

from typing import Dict, Any
import yfinance as yf


def fetch_market_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Fetch recent market data for a given stock ticker.

    Args:
        ticker (str): e.g. "NVDA", "AAPL".
        period (str): e.g. "1mo", "3mo", "1y".
        interval (str): e.g. "1d", "1h".

    Returns:
        Dict[str, Any]: { "ticker": str, "history": [ {date, open, high, low, close, volume}, ...] }
    """
    print(f"[market_api_client] Fetching data for {ticker} ({period}, {interval}) ...")

    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval)

    if df.empty:
        print(f"[market_api_client] ⚠️ No data found for {ticker}.")
        return {"ticker": ticker, "history": []}

    records = []
    for index, row in df.iterrows():
        records.append(
            {
                "date": index.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )

    print(f"[market_api_client] ✅ Retrieved {len(records)} records for {ticker}.")
    return {"ticker": ticker, "history": records}
