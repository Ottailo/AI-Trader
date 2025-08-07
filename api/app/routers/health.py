"""
Health check router for monitoring application status.

This router provides endpoints for checking the health and status
of the AI Trader API and its dependencies.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict:
    """
    Basic health check endpoint.
    
    Returns:
        Dict: Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-trader-api",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict:
    """
    Detailed health check including database connectivity.
    
    Args:
        db: Database session dependency
        
    Returns:
        Dict: Detailed health status information
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-trader-api",
        "version": "1.0.0",
        "checks": {
            "database": "unknown",
            "external_apis": "unknown"
        }
    }
    
    # Check database connectivity
    try:
        # Simple database query to test connectivity
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        health_status["checks"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check external API connectivity (basic check)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test Alpha Vantage API connectivity
            response = await client.get("https://www.alphavantage.co/query", params={"function": "SYMBOL_SEARCH", "keywords": "AAPL", "apikey": "demo"})
            if response.status_code == 200:
                health_status["checks"]["external_apis"] = "healthy"
            else:
                health_status["checks"]["external_apis"] = "degraded"
    except Exception as e:
        logger.error("External API health check failed", error=str(e))
        health_status["checks"]["external_apis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Update overall status
    if any(check == "unhealthy" for check in health_status["checks"].values()):
        health_status["status"] = "unhealthy"
    elif any(check == "degraded" for check in health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/health/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> Dict:
    """
    Readiness check for Kubernetes/container orchestration.
    
    Args:
        db: Database session dependency
        
    Returns:
        Dict: Readiness status
    """
    try:
        # Check database connectivity
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/live")
async def liveness_check() -> Dict:
    """
    Liveness check for Kubernetes/container orchestration.
    
    Returns:
        Dict: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }
