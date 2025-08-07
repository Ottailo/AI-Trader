"""
Health check tests for the AI Trader API.

These tests verify that the basic health endpoints are working correctly.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the basic health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-trader-api"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


def test_liveness_check():
    """Test the liveness check endpoint."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


def test_readiness_check():
    """Test the readiness check endpoint."""
    response = client.get("/api/v1/health/ready")
    # This might fail if database is not available in test environment
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_detailed_health_check():
    """Test the detailed health check endpoint."""
    response = client.get("/api/v1/health/detailed")
    # This might fail if database is not available in test environment
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "timestamp" in data
    assert "service" in data
    assert "version" in data
