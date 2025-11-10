"""
Stub for turning SEC filings into text chunks.
"""

from typing import List, Dict


def build_sec_sentences(filings: List[Dict]) -> List[str]:
    """
    Stub: convert SEC filing metadata into simple sentences.

    Args:
        filings (List[Dict]): Data from fetch_sec_filings() or PDF parsing.

    Returns:
        List[str]: One sentence per filing (stub).
    """
    sentences: List[str] = []

    for f in filings:
        ticker = f.get("ticker", "UNKNOWN")
        form = f.get("form", "FORM")
        year = f.get("year", "YEAR")
        url = f.get("url", "")
        sentence = f"SEC filing for {ticker}: form {form} for year {year}, see {url}."
        sentences.append(sentence)

    return sentences
