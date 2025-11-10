"""
Configuration for rag-api-service.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RagApiSettings:
    # Root of repo: .../advanced-multisource-rag-finance
    base_dir: Path = Path(__file__).resolve().parents[4]

    env: str = "dev"
    default_model_name: str = "llm-stub"
    enable_graph_retrieval: bool = True


def get_settings() -> RagApiSettings:
    return RagApiSettings()
