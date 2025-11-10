"""
Text Cleaning
-------------
This module will handle basic cleanup of raw text:
- stripping extra whitespace
- normalizing newlines
- removing obvious noise (later)
"""

def clean_text(raw_text: str) -> str:
    """
    Very simple placeholder cleaner.

    Args:
        raw_text (str): The original text.

    Returns:
        str: Cleaned text (for now, just stripped).
    """
    cleaned = raw_text.strip()
    # Later we will:
    #  - normalize multiple spaces/newlines
    #  - remove weird characters
    #  - maybe handle headers/footers
    return cleaned
