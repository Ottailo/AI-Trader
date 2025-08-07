"""
Market data service for fetching and processing market data from external APIs.

This service handles data fetching from Alpha Vantage and Polygon.io APIs
with proper error handling, rate limiting, and data validation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging import get_logger, log_market_data_event

logger = get_logger(__name__)


class MarketDataService:
    """Service for fetching market data from external APIs."""
    
    def __init__(self):
        """Initialize the market data service."""
        self.alpha_vantage_api_key = settings.alpha_vantage_api_key
        self.polygon_api_key = settings.polygon_api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_alpha_vantage_data(
        self, 
        symbol: str, 
        interval: str = "5min",
        output_size: str = "compact"
    ) -> Optional[Dict]:
        """
        Fetch market data from Alpha Vantage API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily)
            output_size: Data size (compact=100 data points, full=20+ years)
            
        Returns:
            Dict: Market data response or None if failed
        """
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY" if interval != "daily" else "TIME_SERIES_DAILY",
                "symbol": symbol,
                "interval": interval,
                "outputsize": output_size,
                "apikey": self.alpha_vantage_api_key,
                "datatype": "json"
            }
            
            if interval == "daily":
                params.pop("interval")
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logger.error("Alpha Vantage API error", error=data["Error Message"])
                return None
            
            if "Note" in data:
                logger.warning("Alpha Vantage rate limit", note=data["Note"])
                await asyncio.sleep(60)  # Wait for rate limit
                return None
            
            log_market_data_event(
                logger, 
                "alpha_vantage_data_fetched",
                asset_symbol=symbol,
                interval=interval,
                data_points=len(data.get("Time Series (5min)", {}))
            )
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error fetching Alpha Vantage data", status_code=e.response.status_code)
            raise
        except Exception as e:
            logger.error("Error fetching Alpha Vantage data", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_polygon_data(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        interval: str = "5"
    ) -> Optional[Dict]:
        """
        Fetch market data from Polygon.io API.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            from_date: Start date for data fetch
            to_date: End date for data fetch
            interval: Time interval in minutes (1, 5, 15, 30, 60)
            
        Returns:
            Dict: Market data response or None if failed
        """
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/minute/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}"
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 5000,
                "apikey": self.polygon_api_key
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK":
                logger.error("Polygon API error", status=data.get("status"), error=data.get("error"))
                return None
            
            log_market_data_event(
                logger,
                "polygon_data_fetched",
                asset_symbol=symbol,
                interval=f"{interval}min",
                data_points=len(data.get("results", []))
            )
            
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error fetching Polygon data", status_code=e.response.status_code)
            raise
        except Exception as e:
            logger.error("Error fetching Polygon data", error=str(e))
            raise
    
    def parse_alpha_vantage_data(self, data: Dict, symbol: str) -> List[Tuple]:
        """
        Parse Alpha Vantage API response into structured data.
        
        Args:
            data: Raw API response
            symbol: Stock symbol
            
        Returns:
            List[Tuple]: List of (timestamp, open, high, low, close, volume) tuples
        """
        parsed_data = []
        
        # Determine the time series key based on data structure
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No time series data found in Alpha Vantage response")
            return parsed_data
        
        time_series_data = data[time_series_key]
        
        for timestamp_str, values in time_series_data.items():
            try:
                # Parse timestamp
                if " " in timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                
                # Extract OHLCV data
                open_price = float(values.get("1. open", 0))
                high_price = float(values.get("2. high", 0))
                low_price = float(values.get("3. low", 0))
                close_price = float(values.get("4. close", 0))
                volume = int(values.get("5. volume", 0))
                
                parsed_data.append((
                    timestamp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ))
                
            except (ValueError, KeyError) as e:
                logger.warning("Error parsing Alpha Vantage data point", timestamp=timestamp_str, error=str(e))
                continue
        
        # Sort by timestamp
        parsed_data.sort(key=lambda x: x[0])
        
        return parsed_data
    
    def parse_polygon_data(self, data: Dict, symbol: str) -> List[Tuple]:
        """
        Parse Polygon.io API response into structured data.
        
        Args:
            data: Raw API response
            symbol: Stock symbol
            
        Returns:
            List[Tuple]: List of (timestamp, open, high, low, close, volume) tuples
        """
        parsed_data = []
        
        results = data.get("results", [])
        
        for result in results:
            try:
                # Parse timestamp (milliseconds since epoch)
                timestamp = datetime.fromtimestamp(result["t"] / 1000)
                
                # Extract OHLCV data
                open_price = float(result.get("o", 0))
                high_price = float(result.get("h", 0))
                low_price = float(result.get("l", 0))
                close_price = float(result.get("c", 0))
                volume = int(result.get("v", 0))
                
                parsed_data.append((
                    timestamp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ))
                
            except (ValueError, KeyError) as e:
                logger.warning("Error parsing Polygon data point", result=result, error=str(e))
                continue
        
        return parsed_data
    
    async def get_latest_data(
        self, 
        symbol: str, 
        interval: str = "5min",
        provider: str = "alpha_vantage"
    ) -> Optional[List[Tuple]]:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval
            provider: Data provider (alpha_vantage or polygon)
            
        Returns:
            List[Tuple]: Parsed market data or None if failed
        """
        try:
            if provider == "alpha_vantage":
                data = await self.fetch_alpha_vantage_data(symbol, interval)
                if data:
                    return self.parse_alpha_vantage_data(data, symbol)
            elif provider == "polygon" and self.polygon_api_key:
                # For Polygon, we need to specify a date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  # Get last 7 days
                data = await self.fetch_polygon_data(symbol, start_date, end_date, interval.replace("min", ""))
                if data:
                    return self.parse_polygon_data(data, symbol)
            else:
                logger.error("Unsupported provider or missing API key", provider=provider)
            
            return None
            
        except Exception as e:
            logger.error("Error getting latest data", symbol=symbol, provider=provider, error=str(e))
            return None
