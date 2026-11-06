"""
Google Gemini LLM Provider implementation using the official google-genai SDK.
Supports Google's modern Interactions API (client.interactions.create) as primary engine,
with fallback support for generate_content.
"""
import os
from typing import Any, Optional

import httpx
from google import genai
from google.genai import errors, types

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

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiProvider(BaseLLMProvider):
    """
    Adapter implementing BaseLLMProvider for Google Gemini models via google-genai.
    Utilizes Google's recommended Interactions API as the primary engine.
    """

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        client: Optional[Any] = None,
        use_interactions_api: Optional[bool] = None,
    ) -> None:
        """
        Initialize GeminiProvider.

        Args:
            config: Optional LLMProviderConfig. If omitted, default configuration
                    is derived from environment variables.
            client: Optional pre-configured genai.Client instance (useful for unit testing/mocking).
            use_interactions_api: Whether to use client.interactions.create (True)
                                  or client.models.generate_content (False).
                                  Defaults to True (Google's recommendation for new projects).
        """
        # Resolve model name: explicit config > GEMINI_MODEL env var > DEFAULT_GEMINI_MODEL
        model_name = (
            config.model_name
            if config and config.model_name
            else os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        )
        # Resolve API key: explicit config > GEMINI_API_KEY env var
        api_key = (
            config.api_key
            if config and config.api_key
            else os.getenv("GEMINI_API_KEY", "").strip() or None
        )

        temperature = config.temperature if config else 0.7
        timeout = config.timeout if config else 30.0
        max_output_tokens = config.max_output_tokens if config else None

        self._config = LLMProviderConfig(
            provider_name="gemini",
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
        )
        self._client = client

        if use_interactions_api is not None:
            self.use_interactions_api = use_interactions_api
        else:
            self.use_interactions_api = (
                os.getenv("GEMINI_USE_INTERACTIONS_API", "true").strip().lower() in ("true", "1", "yes")
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
                "Gemini API key is missing or not configured. Set GEMINI_API_KEY in .env.",
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

    def _get_client(self) -> genai.Client:
        """Lazy-initialize and return the genai.Client instance."""
        if self._client is not None:
            return self._client

        self.validate_configuration()

        try:
            # HttpOptions.timeout in google-genai expects milliseconds
            timeout_ms = int(self._config.timeout * 1000)
            http_options = types.HttpOptions(
                timeout=timeout_ms,
            )
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=http_options,
            )
            return self._client
        except Exception as exc:
            sanitized = redact_secrets(str(exc))
            raise LLMConfigurationError(
                f"Failed to initialize Google GenAI Client: {sanitized}",
                provider_name=self.provider_name,
            ) from exc

    def _extract_system_and_conversation(
        self, messages: list[LLMMessage]
    ) -> tuple[Optional[str], list[LLMMessage]]:
        """Separate system instruction from conversation messages."""
        system_parts: list[str] = []
        conversation: list[LLMMessage] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content.strip())
            else:
                conversation.append(msg)

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, conversation

    def _convert_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[list[types.Content], Optional[str]]:
        """
        Convert project LLMMessage instances into google-genai Content structures
        and separate system instruction text.
        """
        system_instruction, conversation = self._extract_system_and_conversation(messages)
        contents: list[types.Content] = []
        for msg in conversation:
            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.content)],
                )
            )
        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=" ")]))
        return contents, system_instruction

    def _format_interactions_input(self, conversation: list[LLMMessage]) -> str:
        """
        Format conversation messages into an input payload for client.interactions.create.
        If single message, returns message content directly.
        If multi-turn, formats turns with canonical role indicators.
        """
        if not conversation:
            return " "
        if len(conversation) == 1:
            return conversation[0].content

        turns = []
        for msg in conversation:
            prefix = "User" if msg.role == MessageRole.USER else "Model"
            turns.append(f"{prefix}: {msg.content}")
        return "\n\n".join(turns)

    def _map_error(self, exc: Exception) -> Exception:
        """Map vendor SDK exceptions to project exception hierarchy."""
        sanitized_msg = redact_secrets(str(exc))
        status_code = getattr(exc, "status_code", getattr(exc, "code", None))

        if status_code in (401, 403):
            return LLMAuthenticationError(sanitized_msg, provider_name=self.provider_name)
        if status_code == 429:
            return LLMRateLimitError(sanitized_msg, provider_name=self.provider_name)
        if status_code in (408, 504):
            return LLMTimeoutError(sanitized_msg, provider_name=self.provider_name)
        if status_code == 400:
            return LLMConfigurationError(sanitized_msg, provider_name=self.provider_name)
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return LLMTimeoutError(f"Request timed out: {sanitized_msg}", provider_name=self.provider_name)
        if isinstance(exc, httpx.NetworkError):
            return LLMProviderError(f"Network error: {sanitized_msg}", provider_name=self.provider_name)
        if isinstance(exc, LLMProviderError):
            return exc
        return LLMProviderError(sanitized_msg, provider_name=self.provider_name)

    def _generate_interactions(
        self, client: Any, request: LLMRequest, system_instruction: Optional[str], conversation: list[LLMMessage]
    ) -> LLMResponse:
        """Execute request via Google's recommended Interactions API (client.interactions.create)."""
        input_text = self._format_interactions_input(conversation)
        timeout_sec = float(request.generation_config.timeout or self._config.timeout)

        gen_config: dict[str, Any] = {}
        max_tokens = request.generation_config.max_output_tokens or self._config.max_output_tokens
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        if request.generation_config.stop_sequences:
            gen_config["stop_sequences"] = request.generation_config.stop_sequences
        if request.generation_config.temperature is not None:
            gen_config["temperature"] = request.generation_config.temperature
        if request.generation_config.top_p is not None:
            gen_config["top_p"] = request.generation_config.top_p

        try:
            interaction = client.interactions.create(
                model=self.model_name,
                input=input_text,
                system_instruction=system_instruction,
                generation_config=gen_config if gen_config else None,
                store=False,  # Enforce stateless execution, no server-side retention
                timeout=timeout_sec,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        content = getattr(interaction, "output_text", None)
        if content is None or not content.strip():
            # Check interaction error report
            errors_list = getattr(interaction, "errors", None)
            if errors_list:
                raise LLMResponseError(
                    f"Interaction failed with errors: {errors_list}",
                    provider_name=self.provider_name,
                )
            status = getattr(interaction, "status", "unknown")
            if status == "incomplete":
                raise LLMResponseError(
                    f"Interaction incomplete without text (status: {status}).",
                    provider_name=self.provider_name,
                )
            raise LLMResponseError(
                f"Gemini Interactions API returned empty output text (status: {status}).",
                provider_name=self.provider_name,
            )

        status_str = str(getattr(interaction, "status", "completed")).lower()
        finish_reason = "length" if status_str == "incomplete" else "stop"

        usage: Optional[LLMUsageMetadata] = None
        usage_obj = getattr(interaction, "usage", None)
        if usage_obj is not None:
            usage = LLMUsageMetadata(
                prompt_tokens=getattr(usage_obj, "total_input_tokens", None),
                completion_tokens=getattr(usage_obj, "total_output_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
            )

        return LLMResponse(
            content=content.strip(),
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage=usage,
            finish_reason=finish_reason,
            raw_metadata={
                "interaction_id": getattr(interaction, "id", None),
                "status": status_str,
            },
        )

    def _generate_models(
        self, client: Any, request: LLMRequest, system_instruction: Optional[str], conversation: list[LLMMessage]
    ) -> LLMResponse:
        """Execute request via client.models.generate_content."""
        contents: list[types.Content] = []
        for msg in conversation:
            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=" ")]))

        timeout_sec = float(request.generation_config.timeout or self._config.timeout)
        timeout_ms = int(timeout_sec * 1000)

        config_kwargs: dict[str, Any] = {
            "temperature": request.generation_config.temperature,
            "http_options": types.HttpOptions(timeout=timeout_ms),
        }
        if request.generation_config.top_p is not None:
            config_kwargs["top_p"] = request.generation_config.top_p
        max_tokens = request.generation_config.max_output_tokens or self._config.max_output_tokens
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if request.generation_config.stop_sequences:
            config_kwargs["stop_sequences"] = request.generation_config.stop_sequences
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        gen_config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        if not getattr(response, "candidates", None):
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, "block_reason", None)
                if block_reason:
                    raise LLMResponseError(
                        f"Prompt blocked by safety filter: {block_reason}",
                        provider_name=self.provider_name,
                    )
            raise LLMResponseError(
                "Gemini returned an empty candidate list.",
                provider_name=self.provider_name,
            )

        candidate = response.candidates[0]
        raw_finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason = (
            str(raw_finish_reason.value if hasattr(raw_finish_reason, "value") else raw_finish_reason or "stop")
            .lower()
        )

        if finish_reason in ("safety", "recitation", "blocklist", "prohibited_content", "spii"):
            raise LLMResponseError(
                f"Gemini response blocked by content policy ({finish_reason}).",
                provider_name=self.provider_name,
            )

        content_text = getattr(response, "text", None)
        if content_text is None or not content_text.strip():
            raise LLMResponseError(
                f"Gemini returned empty or blank text content (finish_reason: {finish_reason}).",
                provider_name=self.provider_name,
            )

        usage: Optional[LLMUsageMetadata] = None
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            usage = LLMUsageMetadata(
                prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", None),
                completion_tokens=getattr(response.usage_metadata, "candidates_token_count", None),
                total_tokens=getattr(response.usage_metadata, "total_token_count", None),
            )

        return LLMResponse(
            content=content_text.strip(),
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage=usage,
            finish_reason=finish_reason,
            raw_metadata={
                "model_version": getattr(response, "model_version", None),
                "response_id": getattr(response, "response_id", None),
            },
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Execute generation request against Google Gemini via google-genai SDK.
        Uses client.interactions.create when use_interactions_api is True,
        falling back to client.models.generate_content otherwise.
        """
        self.validate_configuration()
        client = self._get_client()

        system_instruction, conversation = self._extract_system_and_conversation(request.messages)

        if self.use_interactions_api:
            return self._generate_interactions(client, request, system_instruction, conversation)
        return self._generate_models(client, request, system_instruction, conversation)
