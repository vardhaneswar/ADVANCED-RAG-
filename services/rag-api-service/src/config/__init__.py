"""
Config package for rag-api-service.

Exposes get_settings() so other modules can do:
    from config import get_settings
"""

from .settings import get_settings  # noqa: F401
