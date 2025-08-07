"""
Structured logging configuration for the AI Trader API.

This module sets up structured logging with JSON formatting for
production environments and human-readable formatting for development.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from app.config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        BoundLogger: Configured structured logger
    """
    return structlog.get_logger(name)


def log_trade_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    asset_symbol: str,
    trade_type: str,
    quantity: int,
    price: float,
    **kwargs: Any
) -> None:
    """
    Log a trade event with structured data.
    
    Args:
        logger: Structured logger instance
        event: Event type (e.g., 'trade_executed', 'trade_cancelled')
        asset_symbol: Trading symbol
        trade_type: Buy or sell
        quantity: Number of shares
        price: Execution price
        **kwargs: Additional event data
    """
    logger.info(
        event,
        asset_symbol=asset_symbol,
        trade_type=trade_type,
        quantity=quantity,
        price=price,
        **kwargs
    )


def log_market_data_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    asset_symbol: str,
    interval: str,
    data_points: int,
    **kwargs: Any
) -> None:
    """
    Log a market data event with structured data.
    
    Args:
        logger: Structured logger instance
        event: Event type (e.g., 'data_fetched', 'data_processed')
        asset_symbol: Trading symbol
        interval: Data interval (1min, 5min, etc.)
        data_points: Number of data points
        **kwargs: Additional event data
    """
    logger.info(
        event,
        asset_symbol=asset_symbol,
        interval=interval,
        data_points=data_points,
        **kwargs
    )


def log_prediction_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    asset_symbol: str,
    prediction: str,
    confidence: float,
    **kwargs: Any
) -> None:
    """
    Log a prediction event with structured data.
    
    Args:
        logger: Structured logger instance
        event: Event type (e.g., 'prediction_generated', 'signal_triggered')
        asset_symbol: Trading symbol
        prediction: Predicted direction (buy, sell, hold)
        confidence: Model confidence score
        **kwargs: Additional event data
    """
    logger.info(
        event,
        asset_symbol=asset_symbol,
        prediction=prediction,
        confidence=confidence,
        **kwargs
    )


def log_error_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    error: Exception,
    context: Dict[str, Any] = None,
    **kwargs: Any
) -> None:
    """
    Log an error event with structured data.
    
    Args:
        logger: Structured logger instance
        event: Error event type
        error: Exception that occurred
        context: Additional context data
        **kwargs: Additional error data
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_traceback": getattr(error, "__traceback__", None),
    }
    
    if context:
        error_data.update(context)
    
    logger.error(event, **error_data, **kwargs)
