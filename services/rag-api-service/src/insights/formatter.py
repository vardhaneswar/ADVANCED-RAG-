"""
Insights Formatter (stub)
-------------------------
Right now this just passes the LLM answer through.
Later you can parse out KPIs, trends, sentiment, etc.
"""

def format_insight(raw_answer: str) -> str:
    """
    For now just return the raw answer.

    Args:
        raw_answer (str): LLM answer text.

    Returns:
        str: Possibly formatted answer.
    """
    return raw_answer
