"""
Unit tests for application configuration management.
"""
import os
from config import AppConfig, load_config


def test_default_config():
    """Verify default configuration without explicit environment variables."""
    cfg = AppConfig()
    assert cfg.app_name == "AI Career Intelligence Agent"
    assert isinstance(cfg.gemini_key_configured, bool)
    assert isinstance(cfg.groq_key_configured, bool)


def test_load_config():
    """Verify load_config helper loads and returns AppConfig."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.database_url.startswith("sqlite")


def test_key_detection(monkeypatch):
    """Verify key configuration detection without exposing secrets."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-token")
    monkeypatch.setenv("GROQ_API_KEY", "")

    cfg = load_config()
    assert cfg.gemini_key_configured is True
    assert cfg.groq_key_configured is False


def test_groq_model_configuration(monkeypatch):
    """Verify GROQ_MODEL environment configuration."""
    cfg = AppConfig()
    assert cfg.groq_model == "openai/gpt-oss-120b"
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    cfg_custom = load_config()
    assert cfg_custom.groq_model == "llama-3.1-8b-instant"

