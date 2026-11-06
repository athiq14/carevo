"""
Configuration management for AI Career Intelligence Agent.
Securely loads and validates application configuration without exposing secrets.
"""
from pathlib import Path
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Explicitly load .env from the project root
load_dotenv(dotenv_path=BASE_DIR / ".env")


class AppConfig(BaseModel):
    """Application configuration and environment state."""
    app_name: str = "AI Career Intelligence Agent"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    app_debug: bool = Field(
        default_factory=lambda: os.getenv("APP_DEBUG", "true").strip().lower() in ("true", "1", "yes")
    )
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./career_intelligence.db")
    )
    gemini_key_configured: bool = Field(
        default_factory=lambda: bool(os.getenv("GEMINI_API_KEY", "").strip())
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    )
    groq_key_configured: bool = Field(
        default_factory=lambda: bool(os.getenv("GROQ_API_KEY", "").strip())
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    )


def load_config() -> AppConfig:
    """Safely retrieve validated application configuration."""
    return AppConfig()

