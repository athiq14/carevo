"""
Structured exception hierarchy for LLM providers.
Ensures consistent error handling and automatic secret redaction.
"""
import re
from typing import Optional


def redact_secrets(text: str) -> str:
    """
    Sanitize potential API keys and secrets from error messages and logs.
    Targets common patterns (e.g., AIzaSy..., gsk_..., sk-...) as well as generic hex/alphanumeric tokens.
    """
    if not text:
        return ""
    # Redact Google API key pattern (AIzaSy...)
    sanitized = re.sub(r'AIzaSy[A-Za-z0-9_\-]{15,60}', '[REDACTED_API_KEY]', text)
    # Redact Groq API key pattern (gsk_...)
    sanitized = re.sub(r'gsk_[A-Za-z0-9_\-]{15,80}', '[REDACTED_API_KEY]', sanitized)
    # Redact generic OpenAI / Anthropic-style tokens (sk-...)
    sanitized = re.sub(r'sk-[A-Za-z0-9_\-]{15,80}', '[REDACTED_API_KEY]', sanitized)
    return sanitized


class LLMProviderError(Exception):
    """Base exception for all LLM provider failures."""

    def __init__(self, message: str, provider_name: Optional[str] = None) -> None:
        self.raw_message = message
        self.provider_name = provider_name
        self.message = redact_secrets(message)
        prefix = f"[{provider_name}] " if provider_name else ""
        super().__init__(f"{prefix}{self.message}")


class LLMConfigurationError(LLMProviderError):
    """Raised when provider configuration is invalid or missing required parameters."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication credentials (API keys) are missing, expired, or rejected."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when requests exceed provider rate limits or quotas."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when provider network request exceeds designated timeout."""
    pass


class LLMResponseError(LLMProviderError):
    """Raised when provider returns an unparseable, malformed, or empty response."""
    pass


class LLMFallbackError(LLMProviderError):
    """
    Raised when both primary and secondary LLM providers fail in a fallback chain.
    Preserves sanitized error details for both providers.
    """

    def __init__(
        self,
        primary_provider: str,
        primary_error: Exception,
        secondary_provider: str,
        secondary_error: Exception,
        message: Optional[str] = None,
    ) -> None:
        self.primary_provider = primary_provider
        self.primary_error = primary_error
        self.secondary_provider = secondary_provider
        self.secondary_error = secondary_error

        sanitized_p_err = redact_secrets(str(primary_error))
        sanitized_s_err = redact_secrets(str(secondary_error))

        msg = (
            message
            or f"Fallback chain exhausted. Primary provider '{primary_provider}' failed: {sanitized_p_err} | "
               f"Secondary provider '{secondary_provider}' failed: {sanitized_s_err}"
        )
        super().__init__(message=msg, provider_name=primary_provider)


def is_transient_error(exc: Exception) -> bool:
    """
    Determine whether an LLM provider exception represents a transient failure
    eligible for intelligent fallback.

    Transient failures (eligible for fallback):
        - 429: Rate limit or quota exhaustion
        - 503: Service unavailable / capacity overload
        - 502, 504: Bad gateway or gateway timeout
        - Request timeouts
        - Network connection errors, socket disconnects, peer resets

    Permanent failures (NEVER eligible for fallback):
        - 401, 403: Authentication or permission errors
        - 400: Malformed requests, bad arguments, prompt too long
        - 404: Model not found or deprecated endpoint
        - Content filter blocks, safety violations
        - Malformed responses or empty content errors
        - LLMFallbackError: chain already exhausted
    """
    if exc is None:
        return False

    # 1. Fallback error itself is not transient (chain already exhausted)
    if isinstance(exc, LLMFallbackError):
        return False

    # 2. Known project typed exceptions
    if isinstance(exc, (LLMRateLimitError, LLMTimeoutError)):
        return True
    if isinstance(exc, (LLMAuthenticationError, LLMConfigurationError, LLMResponseError)):
        return False

    # 3. Check class name for common network/timeout types
    type_name = type(exc).__name__
    if any(t in type_name for t in ("Timeout", "TimeOut", "ConnectError", "NetworkError", "ConnectionError")):
        return True

    # 4. Check status code attribute if available on exc or underlying cause
    status_code = getattr(exc, "status_code", getattr(exc, "code", None))
    if status_code is None and getattr(exc, "__cause__", None) is not None:
        status_code = getattr(exc.__cause__, "status_code", getattr(exc.__cause__, "code", None))

    if status_code is not None:
        try:
            status_int = int(status_code)
            if status_int in (429, 502, 503, 504, 408):
                return True
            if status_int in (400, 401, 403, 404, 422):
                return False
        except (ValueError, TypeError):
            pass

    # 5. Check error message text for permanent error indicators first
    msg = str(exc).lower()
    permanent_indicators = (
        "401", "403", "400", "404", "422",
        "unauthorized", "authentication", "api key", "forbidden", "permission denied",
        "bad request", "invalid argument", "model not found", "model_not_found",
        "safety", "harmful", "blocked by safety", "content filter",
        "quota exceeded: plan does not support", "invalid parameter",
    )
    if any(ind in msg for ind in permanent_indicators):
        return False

    # 6. Check error message text for transient error indicators
    transient_indicators = (
        "429", "502", "503", "504", "408",
        "rate limit", "rate_limit", "resource exhausted", "too many requests",
        "service unavailable", "overloaded", "capacity", "temporarily unavailable",
        "timeout", "timed out", "deadline exceeded",
        "connection error", "network error", "connection reset", "connection refused",
        "failed to establish a new connection", "socket",
    )
    if any(ind in msg for ind in transient_indicators):
        return True

    # 7. If exc.__cause__ exists, inspect it recursively
    if getattr(exc, "__cause__", None) is not None and exc.__cause__ is not exc:
        return is_transient_error(exc.__cause__)

    return False


