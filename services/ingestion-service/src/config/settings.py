"""
Configuration for ingestion-service.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class IngestionSettings:
    # Root of the repo (…/advanced-multisource-rag-finance)
    # settings.py is at: .../advanced-multisource-rag-finance/services/ingestion-service/src/config/settings.py
    # parents[4] => advanced-multisource-rag-finance
    base_dir: Path = Path(__file__).resolve().parents[4]

    # Where raw + processed data will live
    data_dir: Path = base_dir / "data"

    # Feature flags
    enable_pdf_ingestion: bool = True
    enable_csv_ingestion: bool = True
    enable_json_ingestion: bool = True

    enable_live_market: bool = True
    enable_live_news: bool = True


def get_settings() -> IngestionSettings:
    """Return default settings. Later you can read env vars, etc."""
    settings = IngestionSettings()
    return settings
