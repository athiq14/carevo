"""
Typed exceptions for structured LLM output parsing, validation, and correction.
Enforces secret redaction and diagnostic traceability.
"""
from typing import Any, Optional

from core.llm.exceptions import LLMProviderError, redact_secrets


class StructuredOutputError(LLMProviderError):
    """Base exception for all structured output parsing and validation failures."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        raw_content: Optional[str] = None,
        attempts: int = 1,
    ) -> None:
        self.raw_content = redact_secrets(raw_content) if raw_content else None
        self.model_name = model_name
        self.attempts = attempts
        super().__init__(message=message, provider_name=provider_name)


class JSONParsingError(StructuredOutputError):
    """Raised when LLM output cannot be parsed into valid, unambiguous JSON."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        raw_content: Optional[str] = None,
        attempts: int = 1,
        parse_error: Optional[Exception] = None,
    ) -> None:
        self.parse_error = parse_error
        super().__init__(
            message=message,
            provider_name=provider_name,
            model_name=model_name,
            raw_content=raw_content,
            attempts=attempts,
        )


class StructuredValidationError(StructuredOutputError):
    """Raised when parsed JSON fails Pydantic schema validation."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list[dict[str, Any]]] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        raw_content: Optional[str] = None,
        attempts: int = 1,
    ) -> None:
        self.validation_errors = validation_errors or []
        super().__init__(
            message=message,
            provider_name=provider_name,
            model_name=model_name,
            raw_content=raw_content,
            attempts=attempts,
        )


class CorrectionExhaustedError(StructuredOutputError):
    """Raised when structured generation retries reach the maximum limit without passing validation."""

    def __init__(
        self,
        message: str,
        last_error: Exception,
        attempts: int = 2,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        raw_content: Optional[str] = None,
    ) -> None:
        self.last_error = last_error
        super().__init__(
            message=message,
            provider_name=provider_name,
            model_name=model_name,
            raw_content=raw_content,
            attempts=attempts,
        )

