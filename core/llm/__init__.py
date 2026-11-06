"""
Core LLM Provider Abstraction Architecture.
Exposes uniform interfaces, models, exceptions, and manager for AI provider integration.
"""
from core.llm.base import BaseLLMProvider
from core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMFallbackError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    is_transient_error,
    redact_secrets,
)
from core.llm.gemini_provider import GeminiProvider
from core.llm.groq_provider import GroqProvider
from core.llm.manager import LLMProviderManager
from core.llm.mock_provider import MockLLMProvider
from core.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageMetadata,
    MessageRole,
)
from core.llm.structured import (
    CorrectionExhaustedError,
    JSONParsingError,
    StructuredOutputEngine,
    StructuredOutputError,
    StructuredOutputRequest,
    StructuredOutputResponse,
    StructuredValidationError,
)

__all__ = [
    "BaseLLMProvider",
    "CorrectionExhaustedError",
    "GeminiProvider",
    "GroqProvider",
    "JSONParsingError",
    "LLMAuthenticationError",
    "LLMConfigurationError",
    "LLMFallbackError",
    "LLMGenerationConfig",
    "LLMMessage",
    "LLMProviderConfig",
    "LLMProviderError",
    "LLMProviderManager",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUsageMetadata",
    "MessageRole",
    "MockLLMProvider",
    "StructuredOutputEngine",
    "StructuredOutputError",
    "StructuredOutputRequest",
    "StructuredOutputResponse",
    "StructuredValidationError",
    "is_transient_error",
    "redact_secrets",
]

