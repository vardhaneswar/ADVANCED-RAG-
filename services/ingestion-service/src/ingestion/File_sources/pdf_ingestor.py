"""
PDF Ingestor (stub)
-------------------
Later this will extract text from PDFs (SEC filings or user uploads).
For now, it only logs the directory.
"""

from pathlib import Path
from typing import List


def ingest_pdfs(pdf_dir: Path) -> List[str]:
    """
    Scan a folder for PDFs and (later) extract text.

    Args:
        pdf_dir (Path): Directory containing PDF files.

    Returns:
        List[str]: One raw text string per PDF (stubbed for now).
    """
    print(f"[pdf_ingestor] Would ingest all PDFs from: {pdf_dir}")
    # TODO: Use pdfplumber / pymupdf to actually extract text.
    return []
