"""
Groq LLM Provider implementation using the official groq Python SDK.
Supports ultra-fast LPU inference across modern models such as openai/gpt-oss-120b.
"""
import os
from typing import Any, Optional

import httpx
from groq import Groq
import groq

from core.llm.base import BaseLLMProvider
from core.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    redact_secrets,
)
from core.llm.models import (
    LLMMessage,
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageMetadata,
    MessageRole,
)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqProvider(BaseLLMProvider):
    """
    Adapter implementing BaseLLMProvider for Groq LPU models via the official groq SDK.
    """

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        client: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """
        Initialize GroqProvider.

        Args:
            config: Optional LLMProviderConfig. If omitted, default configuration
                    is derived from environment variables.
            client: Optional pre-configured groq.Groq client instance (useful for unit testing/mocking).
            reasoning_effort: Optional reasoning effort setting ('low', 'medium', 'high')
                             for supported reasoning models like openai/gpt-oss-120b.
        """
        model_name = (
            config.model_name
            if config and config.model_name
            else os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
        )
        api_key = (
            config.api_key
            if config and config.api_key
            else os.getenv("GROQ_API_KEY", "").strip() or None
        )

        temperature = config.temperature if config else 0.7
        timeout = config.timeout if config else 30.0
        max_output_tokens = config.max_output_tokens if config else None

        self._config = LLMProviderConfig(
            provider_name="groq",
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        self._client = client
        self.reasoning_effort = (
            reasoning_effort
            or os.getenv("GROQ_REASONING_EFFORT", "").strip()
            or None
        )

    @property
    def provider_name(self) -> str:
        return self._config.provider_name

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def config(self) -> LLMProviderConfig:
        return self._config

    @property
    def api_key(self) -> Optional[str]:
        """Access raw API key securely for SDK authentication without exposure."""
        return self._config.api_key

    def is_available(self) -> bool:
        """Indicate whether required credentials are configured."""
        return bool(self.api_key and self.api_key.strip())

    def validate_configuration(self) -> None:
        """
        Validate provider configuration and API key readiness.

        Raises:
            LLMAuthenticationError: If API key is missing or blank.
            LLMConfigurationError: If configuration values (e.g. temperature) are out of bounds.
        """
        if not self.is_available():
            raise LLMAuthenticationError(
                "Groq API key is missing or not configured. Set GROQ_API_KEY in .env.",
                provider_name=self.provider_name,
            )
        if self._config.temperature < 0.0 or self._config.temperature > 2.0:
            raise LLMConfigurationError(
                f"Invalid temperature {self._config.temperature}. Must be between 0.0 and 2.0.",
                provider_name=self.provider_name,
            )
        if self._config.timeout <= 0.0:
            raise LLMConfigurationError(
                f"Invalid timeout {self._config.timeout}. Must be greater than 0.",
                provider_name=self.provider_name,
            )

    def _get_client(self) -> Groq:
        """Lazy-initialize and return the groq.Groq client instance."""
        if self._client is not None:
            return self._client

        self.validate_configuration()

        try:
            self._client = Groq(
                api_key=self.api_key,
                timeout=self._config.timeout,
                max_retries=0,  # Strict single-attempt execution, zero SDK-level retries
            )
            return self._client
        except Exception as exc:
            sanitized = redact_secrets(str(exc))
            raise LLMConfigurationError(
                f"Failed to initialize Groq Client: {sanitized}",
                provider_name=self.provider_name,
            ) from exc

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        """
        Convert project LLMMessage instances into Groq chat completion messages.
        Roles map canonically: SYSTEM -> 'system', USER -> 'user', ASSISTANT -> 'assistant'.
        """
        groq_messages: list[dict[str, str]] = []
        for msg in messages:
            role_str = msg.role.value
            groq_messages.append({"role": role_str, "content": msg.content})

        if not groq_messages:
            groq_messages.append({"role": "user", "content": " "})
        return groq_messages

    def _map_error(self, exc: Exception) -> Exception:
        """Map vendor SDK exceptions to project exception hierarchy."""
        sanitized_msg = redact_secrets(str(exc))
        status_code = getattr(exc, "status_code", getattr(exc, "code", None))

        if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)) or status_code in (401, 403):
            return LLMAuthenticationError(sanitized_msg, provider_name=self.provider_name)
        if isinstance(exc, groq.RateLimitError) or status_code == 429:
            return LLMRateLimitError(sanitized_msg, provider_name=self.provider_name)
        if isinstance(exc, (groq.APITimeoutError, httpx.TimeoutException, TimeoutError)) or status_code in (408, 504):
            return LLMTimeoutError(f"Request timed out: {sanitized_msg}", provider_name=self.provider_name)
        if isinstance(exc, (groq.BadRequestError, groq.UnprocessableEntityError)) or status_code in (400, 422):
            return LLMConfigurationError(sanitized_msg, provider_name=self.provider_name)
        if isinstance(exc, groq.APIResponseValidationError):
            return LLMResponseError(sanitized_msg, provider_name=self.provider_name)
        if isinstance(exc, (groq.APIConnectionError, httpx.NetworkError)):
            return LLMProviderError(f"Network error: {sanitized_msg}", provider_name=self.provider_name)
        if isinstance(exc, LLMProviderError):
            return exc
        return LLMProviderError(sanitized_msg, provider_name=self.provider_name)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute generation request against Groq via the official groq SDK.
        """
        self.validate_configuration()
        client = self._get_client()

        groq_messages = self._convert_messages(request.messages)

        # Timeout in seconds
        timeout_sec = float(request.generation_config.timeout or self._config.timeout)

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": groq_messages,
            "temperature": request.generation_config.temperature,
            "timeout": timeout_sec,
        }

        if request.generation_config.top_p is not None:
            kwargs["top_p"] = request.generation_config.top_p

        max_tokens = request.generation_config.max_output_tokens or self._config.max_output_tokens
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if request.generation_config.stop_sequences:
            kwargs["stop"] = request.generation_config.stop_sequences

        if self.reasoning_effort and self.reasoning_effort.lower() in ("low", "medium", "high"):
            kwargs["reasoning_effort"] = self.reasoning_effort.lower()

        if request.generation_config.response_format is not None:
            kwargs["response_format"] = request.generation_config.response_format

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMResponseError(
                "Groq returned an empty choices list.",
                provider_name=self.provider_name,
            )

        choice = choices[0]
        raw_finish_reason = getattr(choice, "finish_reason", None)
        finish_reason = str(raw_finish_reason).lower() if raw_finish_reason else "stop"

        # Usage accounting and reasoning tokens extraction
        usage: Optional[LLMUsageMetadata] = None
        reasoning_tokens: Optional[int] = None
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            details = getattr(usage_obj, "completion_tokens_details", None)
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", None)

            usage = LLMUsageMetadata(
                prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
                completion_tokens=getattr(usage_obj, "completion_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
                reasoning_tokens=reasoning_tokens,
            )

        raw_metadata: dict[str, Any] = {
            "id": getattr(response, "id", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": finish_reason,
        }
        if reasoning_tokens is not None:
            raw_metadata["reasoning_tokens"] = reasoning_tokens
        x_groq = getattr(response, "x_groq", None)
        if x_groq is not None and hasattr(x_groq, "id"):
            raw_metadata["x_groq_id"] = x_groq.id

        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if content is None or not content.strip():
            if finish_reason == "length":
                reasoning_info = f" ({reasoning_tokens} reasoning tokens consumed)" if reasoning_tokens else ""
                raise LLMResponseError(
                    f"Groq generation reached token limit before visible text was generated "
                    f"(finish_reason: length{reasoning_info}). Increase max_output_tokens.",
                    provider_name=self.provider_name,
                )
            raise LLMResponseError(
                f"Groq returned empty or blank text content (finish_reason: {finish_reason}).",
                provider_name=self.provider_name,
            )

        return LLMResponse(
            content=content.strip(),
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage=usage,
            finish_reason=finish_reason,
            raw_metadata=raw_metadata,
        )

