# """
# Graph Retrieval (stub)
# ----------------------
# Simulates retrieving related entities from a graph DB
# (e.g. Neo4j).
# """

# from typing import List, Dict


# def retrieve_from_graph(query: str) -> List[Dict]:
#     """
#     Pretend to query a graph for relationships relevant to the query.

#     Args:
#         query (str): User question.

#     Returns:
#         List[Dict]: Each dict describes a relationship.
#     """
#     print(f"[graph_retriever] (stub) Retrieving graph context for query: {query!r}")

#     # Example of what a real graph result might look like
#     return [
#         {
#             "head": "NVDA",
#             "relation": "SUPPLIER",
#             "tail": "TSMC",
#         },
#         {
#             "head": "NVDA",
#             "relation": "COMPETITOR",
#             "tail": "AAPL",
#         },
#     ]

#version 2
"""
Graph Retrieval (intelligent stub)
----------------------
Simulates retrieving related entities from a graph DB
(e.g. Neo4j).
Now intelligently generates relationships based on tickers in the query.
"""

from typing import List, Dict
import re


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


def generate_graph_relationships(tickers: List[str]) -> List[Dict]:
    """
    Generate realistic graph relationships for the specified tickers.
    """
    if not tickers:
        return []
    
    # Define relationships between companies
    relationships = {
        'AAPL': [
            {'head': 'AAPL', 'relation': 'SUPPLIER', 'tail': 'TSMC'},
            {'head': 'AAPL', 'relation': 'COMPETITOR', 'tail': 'GOOGL'},
            {'head': 'AAPL', 'relation': 'PARTNER', 'tail': 'MSFT'},
        ],
        'MSFT': [
            {'head': 'MSFT', 'relation': 'COMPETITOR', 'tail': 'GOOGL'},
            {'head': 'MSFT', 'relation': 'PARTNER', 'tail': 'AAPL'},
            {'head': 'MSFT', 'relation': 'ACQUIRER', 'tail': 'ACTIVISION'},
        ],
        'GOOGL': [
            {'head': 'GOOGL', 'relation': 'COMPETITOR', 'tail': 'MSFT'},
            {'head': 'GOOGL', 'relation': 'COMPETITOR', 'tail': 'AMZN'},
            {'head': 'GOOGL', 'relation': 'COMPETITOR', 'tail': 'META'},
        ],
        'AMZN': [
            {'head': 'AMZN', 'relation': 'COMPETITOR', 'tail': 'GOOGL'},
            {'head': 'AMZN', 'relation': 'COMPETITOR', 'tail': 'MSFT'},
            {'head': 'AMZN', 'relation': 'CUSTOMER', 'tail': 'NVDA'},
        ],
        'META': [
            {'head': 'META', 'relation': 'COMPETITOR', 'tail': 'GOOGL'},
            {'head': 'META', 'relation': 'SUPPLIER', 'tail': 'NVDA'},
            {'head': 'META', 'relation': 'PARTNER', 'tail': 'MSFT'},
        ],
        'TSLA': [
            {'head': 'TSLA', 'relation': 'SUPPLIER', 'tail': 'PANASONIC'},
            {'head': 'TSLA', 'relation': 'COMPETITOR', 'tail': 'RIVIAN'},
            {'head': 'TSLA', 'relation': 'CUSTOMER', 'tail': 'NVDA'},
        ],
        'NVDA': [
            {'head': 'NVDA', 'relation': 'SUPPLIER', 'tail': 'TSMC'},
            {'head': 'NVDA', 'relation': 'COMPETITOR', 'tail': 'AMD'},
            {'head': 'NVDA', 'relation': 'COMPETITOR', 'tail': 'INTC'},
        ],
        'AMD': [
            {'head': 'AMD', 'relation': 'SUPPLIER', 'tail': 'TSMC'},
            {'head': 'AMD', 'relation': 'COMPETITOR', 'tail': 'NVDA'},
            {'head': 'AMD', 'relation': 'COMPETITOR', 'tail': 'INTC'},
        ],
        'INTC': [
            {'head': 'INTC', 'relation': 'COMPETITOR', 'tail': 'AMD'},
            {'head': 'INTC', 'relation': 'COMPETITOR', 'tail': 'NVDA'},
            {'head': 'INTC', 'relation': 'MANUFACTURER', 'tail': 'SELF'},
        ],
        'NFLX': [
            {'head': 'NFLX', 'relation': 'COMPETITOR', 'tail': 'DIS'},
            {'head': 'NFLX', 'relation': 'CUSTOMER', 'tail': 'AMZN'},
            {'head': 'NFLX', 'relation': 'PARTNER', 'tail': 'MSFT'},
        ],
    }
    
    # Collect relationships for all detected tickers
    graph_results = []
    for ticker in tickers[:3]:  # Limit to first 3 tickers
        ticker_rels = relationships.get(ticker, [])
        # Add up to 2 relationships per ticker
        graph_results.extend(ticker_rels[:2])
    
    return graph_results


def retrieve_from_graph(query: str) -> List[Dict]:
    """
    Pretend to query a graph for relationships relevant to the query.
    Now generates relationships based on tickers mentioned in the query.

    Args:
        query (str): User question.

    Returns:
        List[Dict]: Each dict describes a relationship.
    """
    print(f"[graph_retriever] (stub) Retrieving graph context for query: {query!r}")
    
    # Extract tickers from query
    tickers = extract_tickers_from_query(query)
    print(f"[graph_retriever] Detected tickers: {tickers}")
    
    # Generate relationships for those tickers
    graph_results = generate_graph_relationships(tickers)
    
    print(f"[graph_retriever] Returning {len(graph_results)} graph relationships")
    return graph_results