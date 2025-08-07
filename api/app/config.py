"""
Configuration management for the AI Trader API.

This module handles all environment variables and application configuration
using Pydantic settings for type safety and validation.
"""

import os
from typing import List

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable validation."""
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    sync_database_url: str = Field(..., env="SYNC_DATABASE_URL")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # External APIs
    alpha_vantage_api_key: str = Field(..., env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")
    
    # Trading Configuration
    alpaca_api_key: str = Field(default="", env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", 
        env="ALPACA_BASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Trading Parameters
    default_position_size: float = Field(default=0.02, env="DEFAULT_POSITION_SIZE")
    default_stop_loss_atr_multiplier: float = Field(
        default=2.0, 
        env="DEFAULT_STOP_LOSS_ATR_MULTIPLIER"
    )
    default_take_profit_atr_multiplier: float = Field(
        default=3.0, 
        env="DEFAULT_TAKE_PROFIT_ATR_MULTIPLIER"
    )
    max_positions: int = Field(default=5, env="MAX_POSITIONS")
    
    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string to list if needed."""
        if isinstance(v, str):
            # Remove brackets and split by comma
            v = v.strip("[]").split(",")
            return [origin.strip().strip('"').strip("'") for origin in v]
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        """Ensure secret key is secure in production."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ["development", "dev", "local"]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ["production", "prod"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
