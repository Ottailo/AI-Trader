"""
Data ingestion service for storing market data in the database.

This service handles the storage and management of historical market data,
including data validation, deduplication, and batch operations.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Tuple, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.logging import get_logger, log_market_data_event
from app.models.asset import Asset
from app.models.historical_data import HistoricalData
from app.services.market_data import MarketDataService

logger = get_logger(__name__)


class DataIngestionService:
    """Service for ingesting and storing market data."""
    
    def __init__(self, db_session: AsyncSession):
        """Initialize the data ingestion service."""
        self.db = db_session
        self.market_data_service = MarketDataService()
    
    async def get_or_create_asset(
        self, 
        symbol: str, 
        name: str = None, 
        exchange: str = "NASDAQ",
        asset_type: str = "stock"
    ) -> Asset:
        """
        Get an existing asset or create a new one.
        
        Args:
            symbol: Stock symbol
            name: Company name
            exchange: Exchange name
            asset_type: Asset type (stock, etf, crypto, etc.)
            
        Returns:
            Asset: Database asset record
        """
        # Try to find existing asset
        stmt = select(Asset).where(Asset.symbol == symbol.upper())
        result = await self.db.execute(stmt)
        asset = result.scalar_one_or_none()
        
        if asset:
            return asset
        
        # Create new asset
        asset = Asset(
            symbol=symbol.upper(),
            name=name or symbol.upper(),
            exchange=exchange,
            asset_type=asset_type,
            is_active="Y"
        )
        
        self.db.add(asset)
        await self.db.commit()
        await self.db.refresh(asset)
        
        logger.info("Created new asset", symbol=symbol, name=name)
        return asset
    
    async def store_historical_data(
        self, 
        asset_id: int, 
        data_points: List[Tuple], 
        interval: str = "5min"
    ) -> int:
        """
        Store historical market data in the database.
        
        Args:
            asset_id: Database asset ID
            data_points: List of (timestamp, open, high, low, close, volume) tuples
            interval: Time interval for the data
            
        Returns:
            int: Number of data points stored
        """
        stored_count = 0
        
        for timestamp, open_price, high_price, low_price, close_price, volume in data_points:
            try:
                # Check if data point already exists
                stmt = select(HistoricalData).where(
                    HistoricalData.asset_id == asset_id,
                    HistoricalData.timestamp == timestamp,
                    HistoricalData.interval == interval
                )
                result = await self.db.execute(stmt)
                existing_data = result.scalar_one_or_none()
                
                if existing_data:
                    # Update existing data point
                    existing_data.open_price = Decimal(str(open_price))
                    existing_data.high_price = Decimal(str(high_price))
                    existing_data.low_price = Decimal(str(low_price))
                    existing_data.close_price = Decimal(str(close_price))
                    existing_data.volume = volume
                else:
                    # Create new data point
                    historical_data = HistoricalData(
                        asset_id=asset_id,
                        timestamp=timestamp,
                        open_price=Decimal(str(open_price)),
                        high_price=Decimal(str(high_price)),
                        low_price=Decimal(str(low_price)),
                        close_price=Decimal(str(close_price)),
                        volume=volume,
                        interval=interval
                    )
                    self.db.add(historical_data)
                
                stored_count += 1
                
            except Exception as e:
                logger.error(
                    "Error storing historical data point",
                    asset_id=asset_id,
                    timestamp=timestamp,
                    error=str(e)
                )
                continue
        
        # Commit all changes
        await self.db.commit()
        
        log_market_data_event(
            logger,
            "historical_data_stored",
            asset_id=asset_id,
            interval=interval,
            data_points=stored_count
        )
        
        return stored_count
    
    async def fetch_and_store_data(
        self, 
        symbol: str, 
        interval: str = "5min",
        provider: str = "alpha_vantage"
    ) -> Optional[int]:
        """
        Fetch market data and store it in the database.
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            provider: Data provider
            
        Returns:
            int: Number of data points stored or None if failed
        """
        try:
            # Get or create asset
            asset = await self.get_or_create_asset(symbol)
            
            # Fetch market data
            async with self.market_data_service:
                data_points = await self.market_data_service.get_latest_data(
                    symbol, interval, provider
                )
            
            if not data_points:
                logger.warning("No data points received", symbol=symbol, provider=provider)
                return None
            
            # Store data in database
            stored_count = await self.store_historical_data(
                asset.id, data_points, interval
            )
            
            logger.info(
                "Successfully fetched and stored market data",
                symbol=symbol,
                interval=interval,
                provider=provider,
                stored_count=stored_count
            )
            
            return stored_count
            
        except Exception as e:
            logger.error(
                "Error in fetch_and_store_data",
                symbol=symbol,
                interval=interval,
                provider=provider,
                error=str(e)
            )
            return None
    
    async def get_latest_data_from_db(
        self, 
        symbol: str, 
        interval: str = "5min",
        limit: int = 100
    ) -> List[HistoricalData]:
        """
        Get the latest historical data from the database.
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            limit: Maximum number of data points to return
            
        Returns:
            List[HistoricalData]: List of historical data records
        """
        try:
            # Get asset
            stmt = select(Asset).where(Asset.symbol == symbol.upper())
            result = await self.db.execute(stmt)
            asset = result.scalar_one_or_none()
            
            if not asset:
                logger.warning("Asset not found", symbol=symbol)
                return []
            
            # Get latest data
            stmt = (
                select(HistoricalData)
                .where(
                    HistoricalData.asset_id == asset.id,
                    HistoricalData.interval == interval
                )
                .order_by(HistoricalData.timestamp.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            data_points = result.scalars().all()
            
            # Return in chronological order
            return list(reversed(data_points))
            
        except Exception as e:
            logger.error(
                "Error getting latest data from database",
                symbol=symbol,
                interval=interval,
                error=str(e)
            )
            return []
    
    async def get_data_range(
        self, 
        symbol: str, 
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min"
    ) -> List[HistoricalData]:
        """
        Get historical data for a specific date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Time interval
            
        Returns:
            List[HistoricalData]: List of historical data records
        """
        try:
            # Get asset
            stmt = select(Asset).where(Asset.symbol == symbol.upper())
            result = await self.db.execute(stmt)
            asset = result.scalar_one_or_none()
            
            if not asset:
                logger.warning("Asset not found", symbol=symbol)
                return []
            
            # Get data for date range
            stmt = (
                select(HistoricalData)
                .where(
                    HistoricalData.asset_id == asset.id,
                    HistoricalData.interval == interval,
                    HistoricalData.timestamp >= start_date,
                    HistoricalData.timestamp <= end_date
                )
                .order_by(HistoricalData.timestamp.asc())
            )
            
            result = await self.db.execute(stmt)
            data_points = result.scalars().all()
            
            return list(data_points)
            
        except Exception as e:
            logger.error(
                "Error getting data range from database",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                error=str(e)
            )
            return []
    
    async def update_asset_data(
        self, 
        symbols: List[str], 
        interval: str = "5min",
        provider: str = "alpha_vantage"
    ) -> Dict[str, int]:
        """
        Update market data for multiple assets.
        
        Args:
            symbols: List of stock symbols
            interval: Time interval
            provider: Data provider
            
        Returns:
            Dict[str, int]: Dictionary mapping symbols to stored data counts
        """
        results = {}
        
        for symbol in symbols:
            try:
                stored_count = await self.fetch_and_store_data(symbol, interval, provider)
                results[symbol] = stored_count or 0
            except Exception as e:
                logger.error(
                    "Error updating asset data",
                    symbol=symbol,
                    error=str(e)
                )
                results[symbol] = 0
        
        logger.info(
            "Completed batch data update",
            symbols=list(results.keys()),
            results=results
        )
        
        return results
