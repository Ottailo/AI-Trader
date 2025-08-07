"""
Market data router for data ingestion and retrieval.

This router provides endpoints for fetching market data from external APIs,
storing it in the database, and retrieving historical data.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.logging import get_logger
from app.services.data_ingestion import DataIngestionService

logger = get_logger(__name__)
router = APIRouter()


class MarketDataResponse(BaseModel):
    """Pydantic model for market data responses."""
    timestamp: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    interval: str
    
    class Config:
        from_attributes = True


class DataIngestionRequest(BaseModel):
    """Pydantic model for data ingestion requests."""
    symbol: str
    interval: str = "5min"
    provider: str = "alpha_vantage"


class BatchDataIngestionRequest(BaseModel):
    """Pydantic model for batch data ingestion requests."""
    symbols: List[str]
    interval: str = "5min"
    provider: str = "alpha_vantage"


@router.post("/market-data/ingest")
async def ingest_market_data(
    request: DataIngestionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Ingest market data for a specific symbol.
    
    Args:
        request: Data ingestion request
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        dict: Ingestion status
    """
    try:
        ingestion_service = DataIngestionService(db)
        
        # Run ingestion in background
        background_tasks.add_task(
            ingestion_service.fetch_and_store_data,
            request.symbol,
            request.interval,
            request.provider
        )
        
        logger.info(
            "Market data ingestion started",
            symbol=request.symbol,
            interval=request.interval,
            provider=request.provider
        )
        
        return {
            "message": f"Data ingestion started for {request.symbol}",
            "symbol": request.symbol,
            "interval": request.interval,
            "provider": request.provider,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(
            "Error starting data ingestion",
            symbol=request.symbol,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/market-data/ingest/batch")
async def batch_ingest_market_data(
    request: BatchDataIngestionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Ingest market data for multiple symbols.
    
    Args:
        request: Batch data ingestion request
        background_tasks: FastAPI background tasks
        db: Database session
        
    Returns:
        dict: Batch ingestion status
    """
    try:
        ingestion_service = DataIngestionService(db)
        
        # Run batch ingestion in background
        background_tasks.add_task(
            ingestion_service.update_asset_data,
            request.symbols,
            request.interval,
            request.provider
        )
        
        logger.info(
            "Batch market data ingestion started",
            symbols=request.symbols,
            interval=request.interval,
            provider=request.provider
        )
        
        return {
            "message": f"Batch data ingestion started for {len(request.symbols)} symbols",
            "symbols": request.symbols,
            "interval": request.interval,
            "provider": request.provider,
            "status": "started"
        }
        
    except Exception as e:
        logger.error(
            "Error starting batch data ingestion",
            symbols=request.symbols,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-data/{symbol}", response_model=List[MarketDataResponse])
async def get_market_data(
    symbol: str,
    interval: str = Query("5min", description="Time interval"),
    limit: int = Query(100, ge=1, le=1000, description="Number of data points"),
    db: AsyncSession = Depends(get_db)
) -> List[MarketDataResponse]:
    """
    Get historical market data for a symbol.
    
    Args:
        symbol: Stock symbol
        interval: Time interval
        limit: Maximum number of data points
        db: Database session
        
    Returns:
        List[MarketDataResponse]: Historical market data
        
    Raises:
        HTTPException: If symbol not found or no data available
    """
    try:
        ingestion_service = DataIngestionService(db)
        data_points = await ingestion_service.get_latest_data_from_db(
            symbol, interval, limit
        )
        
        if not data_points:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {symbol} with interval {interval}"
            )
        
        return [MarketDataResponse.from_orm(data) for data in data_points]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting market data",
            symbol=symbol,
            interval=interval,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-data/{symbol}/range", response_model=List[MarketDataResponse])
async def get_market_data_range(
    symbol: str,
    start_date: datetime = Query(..., description="Start date"),
    end_date: datetime = Query(..., description="End date"),
    interval: str = Query("5min", description="Time interval"),
    db: AsyncSession = Depends(get_db)
) -> List[MarketDataResponse]:
    """
    Get historical market data for a specific date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        interval: Time interval
        db: Database session
        
    Returns:
        List[MarketDataResponse]: Historical market data for the date range
        
    Raises:
        HTTPException: If symbol not found or no data available
    """
    try:
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        if end_date - start_date > timedelta(days=365):
            raise HTTPException(
                status_code=400,
                detail="Date range cannot exceed 1 year"
            )
        
        ingestion_service = DataIngestionService(db)
        data_points = await ingestion_service.get_data_range(
            symbol, start_date, end_date, interval
        )
        
        if not data_points:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {symbol} in the specified date range"
            )
        
        return [MarketDataResponse.from_orm(data) for data in data_points]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting market data range",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-data/{symbol}/latest")
async def get_latest_market_data(
    symbol: str,
    interval: str = Query("5min", description="Time interval"),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get the latest market data point for a symbol.
    
    Args:
        symbol: Stock symbol
        interval: Time interval
        db: Database session
        
    Returns:
        dict: Latest market data point
        
    Raises:
        HTTPException: If symbol not found or no data available
    """
    try:
        ingestion_service = DataIngestionService(db)
        data_points = await ingestion_service.get_latest_data_from_db(
            symbol, interval, 1
        )
        
        if not data_points:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {symbol}"
            )
        
        latest_data = data_points[0]
        
        return {
            "symbol": symbol,
            "timestamp": latest_data.timestamp.isoformat(),
            "open_price": float(latest_data.open_price),
            "high_price": float(latest_data.high_price),
            "low_price": float(latest_data.low_price),
            "close_price": float(latest_data.close_price),
            "volume": latest_data.volume,
            "interval": latest_data.interval,
            "price_change": float(latest_data.price_change),
            "price_change_percent": float(latest_data.price_change_percent)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting latest market data",
            symbol=symbol,
            interval=interval,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/market-data/{symbol}/summary")
async def get_market_data_summary(
    symbol: str,
    interval: str = Query("5min", description="Time interval"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get a summary of market data for a symbol.
    
    Args:
        symbol: Stock symbol
        interval: Time interval
        days: Number of days to analyze
        db: Database session
        
    Returns:
        dict: Market data summary
        
    Raises:
        HTTPException: If symbol not found or no data available
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ingestion_service = DataIngestionService(db)
        data_points = await ingestion_service.get_data_range(
            symbol, start_date, end_date, interval
        )
        
        if not data_points:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {symbol} in the last {days} days"
            )
        
        # Calculate summary statistics
        prices = [float(data.close_price) for data in data_points]
        volumes = [data.volume for data in data_points]
        
        current_price = prices[-1]
        min_price = min(prices)
        max_price = max(prices)
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        price_change = current_price - prices[0]
        price_change_percent = (price_change / prices[0]) * 100 if prices[0] > 0 else 0
        
        return {
            "symbol": symbol,
            "interval": interval,
            "period_days": days,
            "data_points": len(data_points),
            "current_price": current_price,
            "min_price": min_price,
            "max_price": max_price,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "avg_volume": avg_volume,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error getting market data summary",
            symbol=symbol,
            interval=interval,
            days=days,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")
