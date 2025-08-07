"""
Assets router for managing trading assets.

This router provides endpoints for creating, reading, updating, and deleting
trading assets (stocks, ETFs, etc.) in the system.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.logging import get_logger
from app.models.asset import Asset

logger = get_logger(__name__)
router = APIRouter()


class AssetCreate(BaseModel):
    """Pydantic model for creating assets."""
    symbol: str
    name: str
    exchange: str = "NASDAQ"
    asset_type: str = "stock"
    description: Optional[str] = None


class AssetResponse(BaseModel):
    """Pydantic model for asset responses."""
    id: int
    symbol: str
    name: str
    exchange: str
    asset_type: str
    description: Optional[str] = None
    is_active: str
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


class AssetUpdate(BaseModel):
    """Pydantic model for updating assets."""
    name: Optional[str] = None
    exchange: Optional[str] = None
    asset_type: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[str] = None


@router.post("/assets", response_model=AssetResponse)
async def create_asset(
    asset_data: AssetCreate,
    db: AsyncSession = Depends(get_db)
) -> AssetResponse:
    """
    Create a new trading asset.
    
    Args:
        asset_data: Asset creation data
        db: Database session
        
    Returns:
        AssetResponse: Created asset information
        
    Raises:
        HTTPException: If asset already exists or validation fails
    """
    try:
        # Check if asset already exists
        stmt = select(Asset).where(Asset.symbol == asset_data.symbol.upper())
        result = await db.execute(stmt)
        existing_asset = result.scalar_one_or_none()
        
        if existing_asset:
            raise HTTPException(
                status_code=400,
                detail=f"Asset with symbol {asset_data.symbol} already exists"
            )
        
        # Create new asset
        asset = Asset(
            symbol=asset_data.symbol.upper(),
            name=asset_data.name,
            exchange=asset_data.exchange,
            asset_type=asset_data.asset_type,
            description=asset_data.description,
            is_active="Y"
        )
        
        db.add(asset)
        await db.commit()
        await db.refresh(asset)
        
        logger.info("Asset created", symbol=asset.symbol, name=asset.name)
        
        return AssetResponse.from_orm(asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating asset", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/assets", response_model=List[AssetResponse])
async def list_assets(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    active_only: bool = Query(True, description="Return only active assets"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    db: AsyncSession = Depends(get_db)
) -> List[AssetResponse]:
    """
    List trading assets with optional filtering.
    
    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        active_only: Return only active assets
        asset_type: Filter by asset type
        exchange: Filter by exchange
        db: Database session
        
    Returns:
        List[AssetResponse]: List of assets
    """
    try:
        # Build query
        stmt = select(Asset)
        
        if active_only:
            stmt = stmt.where(Asset.is_active == "Y")
        
        if asset_type:
            stmt = stmt.where(Asset.asset_type == asset_type)
        
        if exchange:
            stmt = stmt.where(Asset.exchange == exchange)
        
        # Add pagination
        stmt = stmt.offset(skip).limit(limit)
        
        result = await db.execute(stmt)
        assets = result.scalars().all()
        
        return [AssetResponse.from_orm(asset) for asset in assets]
        
    except Exception as e:
        logger.error("Error listing assets", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/assets/{symbol}", response_model=AssetResponse)
async def get_asset(
    symbol: str,
    db: AsyncSession = Depends(get_db)
) -> AssetResponse:
    """
    Get a specific asset by symbol.
    
    Args:
        symbol: Asset symbol
        db: Database session
        
    Returns:
        AssetResponse: Asset information
        
    Raises:
        HTTPException: If asset not found
    """
    try:
        stmt = select(Asset).where(Asset.symbol == symbol.upper())
        result = await db.execute(stmt)
        asset = result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with symbol {symbol} not found"
            )
        
        return AssetResponse.from_orm(asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting asset", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/assets/{symbol}", response_model=AssetResponse)
async def update_asset(
    symbol: str,
    asset_data: AssetUpdate,
    db: AsyncSession = Depends(get_db)
) -> AssetResponse:
    """
    Update an existing asset.
    
    Args:
        symbol: Asset symbol
        asset_data: Asset update data
        db: Database session
        
    Returns:
        AssetResponse: Updated asset information
        
    Raises:
        HTTPException: If asset not found
    """
    try:
        stmt = select(Asset).where(Asset.symbol == symbol.upper())
        result = await db.execute(stmt)
        asset = result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with symbol {symbol} not found"
            )
        
        # Update fields
        update_data = asset_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(asset, field, value)
        
        await db.commit()
        await db.refresh(asset)
        
        logger.info("Asset updated", symbol=asset.symbol, updates=update_data)
        
        return AssetResponse.from_orm(asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating asset", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/assets/{symbol}")
async def delete_asset(
    symbol: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Delete an asset (soft delete by setting is_active to 'N').
    
    Args:
        symbol: Asset symbol
        db: Database session
        
    Returns:
        dict: Deletion confirmation
        
    Raises:
        HTTPException: If asset not found
    """
    try:
        stmt = select(Asset).where(Asset.symbol == symbol.upper())
        result = await db.execute(stmt)
        asset = result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with symbol {symbol} not found"
            )
        
        # Soft delete
        asset.is_active = "N"
        await db.commit()
        
        logger.info("Asset deleted", symbol=asset.symbol)
        
        return {"message": f"Asset {symbol} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting asset", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/assets/{symbol}/status")
async def get_asset_status(
    symbol: str,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get the status of an asset including basic information.
    
    Args:
        symbol: Asset symbol
        db: Database session
        
    Returns:
        dict: Asset status information
        
    Raises:
        HTTPException: If asset not found
    """
    try:
        stmt = select(Asset).where(Asset.symbol == symbol.upper())
        result = await db.execute(stmt)
        asset = result.scalar_one_or_none()
        
        if not asset:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with symbol {symbol} not found"
            )
        
        return {
            "symbol": asset.symbol,
            "name": asset.name,
            "is_active": asset.is_active == "Y",
            "is_tradeable": asset.is_tradeable,
            "exchange": asset.exchange,
            "asset_type": asset.asset_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting asset status", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
