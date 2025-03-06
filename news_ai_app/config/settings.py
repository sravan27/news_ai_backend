"""
Configuration settings for the news_ai_app.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class APIKeys(BaseModel):
    """API keys configuration."""
    news_api_key: str = Field(default_factory=lambda: os.getenv('NEWS_API_KEY', ''))
    openai_api_key: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))


class DatabaseSettings(BaseModel):
    """Database configuration."""
    url: str = Field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///./news_ai_app.db'))


class AppSettings(BaseModel):
    """Application configuration."""
    env: str = Field(default_factory=lambda: os.getenv('APP_ENV', 'development'))
    debug: bool = Field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    port: int = Field(default_factory=lambda: int(os.getenv('PORT', '8000')))


class ModelSettings(BaseModel):
    """Model configuration."""
    mind_dataset_path: Path = Field(
        default_factory=lambda: Path(os.getenv('MIND_DATASET_PATH', './MINDLarge'))
    )
    silicon_models_path: Path = Field(
        default_factory=lambda: Path(os.getenv('SILICON_MODELS_PATH', './ml_pipeline/models/deployed'))
    )


class MetricsSettings(BaseModel):
    """Metrics calculation configuration."""
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv('METRICS_CALCULATION_BATCH_SIZE', '64'))
    )


class Settings(BaseModel):
    """Global application settings."""
    api_keys: APIKeys = Field(default_factory=APIKeys)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)


# Create global settings instance
settings = Settings()