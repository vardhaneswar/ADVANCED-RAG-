# """
# Vector Retrieval (stub)
# -----------------------
# Simulates retrieving documents from a vector database
# (e.g. Pinecone) given a query.
# """

# from dataclasses import dataclass
# from typing import List


# @dataclass
# class RetrievedDoc:
#     text: str
#     score: float
#     source: str   # e.g. "market", "news", "sec"


# def retrieve_from_vector_index(query: str, top_k: int = 3) -> List[RetrievedDoc]:
#     """
#     Pretend to query a vector DB and return top_k documents.

#     Args:
#         query (str): User question.
#         top_k (int): Number of docs to return.

#     Returns:
#         List[RetrievedDoc]
#     """
#     print(f"[vector_retriever] (stub) Retrieving top {top_k} docs for query: {query!r}")

#     # These are just placeholder docs. Later these would be
#     # real chunks coming from Pinecone populated by ingestion-service.
#     fake_docs = [
#         RetrievedDoc(
#             text="NVDA has shown strong growth in AI-related GPU demand recently.",
#             score=0.91,
#             source="market",
#         ),
#         RetrievedDoc(
#             text="AAPL revenue is heavily driven by iPhone and services segments.",
#             score=0.88,
#             source="fundamentals",
#         ),
#         RetrievedDoc(
#             text="Recent volatility in NVDA stock is correlated with broader tech sentiment.",
#             score=0.83,
#             source="news",
#         ),
#     ]

#     return fake_docs[:top_k]

# VERSION 2
"""
Vector Retrieval (intelligent stub)
-----------------------
Simulates retrieving documents from a vector database
(e.g. Pinecone) given a query.
Now intelligently generates context based on tickers mentioned in the query.
"""

from dataclasses import dataclass
from typing import List
import re


@dataclass
class RetrievedDoc:
    text: str
    score: float
    source: str   # e.g. "market", "news", "sec"


def extract_tickers_from_query(query: str) -> List[str]:
    """
    Extract ticker symbols from the query.
    Looks for common patterns and known ticker symbols.
    """
    # Common tickers to look for
    common_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
        'AMD', 'INTC', 'NFLX', 'DIS', 'V', 'MA', 'JPM', 'BAC', 'WMT',
        'ORCL', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'UBER', 'LYFT', 'SNAP'
    ]
    
    # Convert query to uppercase for matching
    query_upper = query.upper()
    
    # Find all tickers mentioned in the query
    found_tickers = []
    for ticker in common_tickers:
        # Look for ticker as whole word
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, query_upper):
            found_tickers.append(ticker)
    
    return found_tickers


def generate_context_for_tickers(tickers: List[str]) -> List[RetrievedDoc]:
    """
    Generate realistic context documents for the specified tickers.
    """
    if not tickers:
        # Default fallback
        return [
            RetrievedDoc(
                text="General market conditions show mixed performance across tech sector.",
                score=0.85,
                source="market",
            )
        ]
    
    docs = []
    
    # Generate context for each ticker
    ticker_contexts = {
        'AAPL': [
            "AAPL revenue is heavily driven by iPhone and services segments, showing steady growth in ecosystem subscriptions.",
            "Apple's recent performance indicates strong demand for premium devices despite economic headwinds.",
            "AAPL maintains high margins through its integrated hardware-software ecosystem and strong brand loyalty.",
        ],
        'MSFT': [
            "MSFT cloud revenue (Azure) continues to grow at double-digit rates, driven by enterprise adoption.",
            "Microsoft's AI investments in OpenAI partnership are expected to drive future growth opportunities.",
            "MSFT shows strong fundamentals with diversified revenue streams across cloud, productivity, and gaming.",
        ],
        'GOOGL': [
            "GOOGL advertising revenue remains strong despite competition, with YouTube showing particular strength.",
            "Google Cloud is gaining market share in enterprise cloud infrastructure market.",
            "GOOGL faces regulatory scrutiny but maintains dominant position in search and digital advertising.",
        ],
        'AMZN': [
            "AMZN AWS cloud services remain the primary profit driver despite e-commerce being larger by revenue.",
            "Amazon's e-commerce business shows resilient growth with improving margins in North America.",
            "AMZN continues to invest heavily in logistics infrastructure and AI capabilities for long-term growth.",
        ],
        'META': [
            "META advertising revenue rebounded after efficiency improvements and significant cost cuts in 2023.",
            "Meta's Reality Labs division (VR/AR) continues to show losses but represents long-term strategic bet.",
            "META user growth remains strong across Facebook, Instagram, and WhatsApp platforms globally.",
        ],
        'TSLA': [
            "TSLA deliveries show strong growth despite increased competition in the global EV market.",
            "Tesla's energy storage business is becoming an increasingly important secondary revenue stream.",
            "TSLA maintains technology lead in autonomous driving, though regulatory approval remains uncertain.",
        ],
        'NVDA': [
            "NVDA has shown exceptional growth in AI-related GPU demand, driven by data center expansion.",
            "Nvidia's data center revenue has surpassed gaming as the primary business segment in recent quarters.",
            "NVDA stock volatility is correlated with broader AI and tech sentiment shifts in the market.",
        ],
        'AMD': [
            "AMD gains market share in both CPU and GPU markets with competitive price-performance offerings.",
            "AMD's data center business shows strong growth competing with Intel and Nvidia in key segments.",
            "AMD benefits from TSMC's advanced manufacturing while competitors face foundry constraints.",
        ],
        'INTC': [
            "INTC struggles with manufacturing delays but shows progress in process technology roadmap recovery.",
            "Intel's foundry services strategy aims to compete with TSMC in coming years through IFS division.",
            "INTC maintains strong position in data center CPUs despite increasing AMD competition.",
        ],
        'NFLX': [
            "NFLX subscriber growth stabilized after introducing ad-supported tier and password sharing crackdown.",
            "Netflix maintains content spending leadership but faces increased competition from Disney+ and others.",
            "NFLX shows improving free cash flow generation as content library matures and efficiency improves.",
        ],
    }
    
    # Add documents for each ticker found
    for i, ticker in enumerate(tickers[:3]):  # Limit to first 3 tickers
        contexts = ticker_contexts.get(ticker, [
            f"{ticker} shows typical market performance with standard volatility patterns and sector correlation."
        ])
        
        # Add 2 documents per ticker for richer context
        for j, context in enumerate(contexts[:2]):
            score = 0.92 - (i * 0.05) - (j * 0.03)  # Decreasing scores
            source = ["market", "fundamentals", "news"][j % 3]
            docs.append(RetrievedDoc(text=context, score=score, source=source))
    
    return docs


def retrieve_from_vector_index(query: str, top_k: int = 3) -> List[RetrievedDoc]:
    """
    Pretend to query a vector DB and return top_k documents.
    Now generates context based on tickers mentioned in the query.

    Args:
        query (str): User question.
        top_k (int): Number of docs to return.

    Returns:
        List[RetrievedDoc]
    """
    print(f"[vector_retriever] (stub) Retrieving top {top_k} docs for query: {query!r}")
    
    # Extract tickers from query
    tickers = extract_tickers_from_query(query)
    print(f"[vector_retriever] Detected tickers: {tickers}")
    
    # Generate context for those tickers
    fake_docs = generate_context_for_tickers(tickers)
    
    # Return top_k documents
    return fake_docs[:top_k]