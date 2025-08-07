"""
API routers package for the AI Trader API.

This package contains all FastAPI routers for different API endpoints.
"""

from . import assets, health, market_data, trading, ml

__all__ = ["assets", "health", "market_data", "trading", "ml"]
