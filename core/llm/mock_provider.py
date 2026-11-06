"""
Deterministic, offline Mock LLM Provider implementing BaseLLMProvider.
Requires 0 API keys and generates 0 network calls.
Ideal for testing, CI/CD, local development, and error simulation.
"""
from typing import Optional

from core.llm.base import BaseLLMProvider
from core.llm.exceptions import LLMProviderError, LLMResponseError
from core.llm.models import (
    LLMProviderConfig,
    LLMRequest,
    LLMResponse,
    LLMUsageMetadata,
)


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider for deterministic offline testing and error simulation.
    """

    def __init__(
        self,
        config: Optional[LLMProviderConfig] = None,
        canned_response: Optional[str] = None,
        simulate_error: Optional[Exception] = None,
        is_available_override: Optional[bool] = None,
    ) -> None:
        """
        Initialize MockLLMProvider.

        Args:
            config: Optional LLMProviderConfig. If omitted, a default mock config is created.
            canned_response: Optional explicit string to return on every generate() call.
            simulate_error: Optional exception to raise when generate() or validate_configuration() is called.
            is_available_override: Optional explicit boolean returned by is_available().
        """
        self._config = config or LLMProviderConfig(
            provider_name="mock",
            model_name="mock-career-v1",
            api_key=None,
            temperature=0.0,
            timeout=5.0,
        )
        self.canned_response = canned_response
        self.simulate_error = simulate_error
        self.is_available_override = is_available_override
        self.call_history: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return self._config.provider_name

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def config(self) -> LLMProviderConfig:
        return self._config

    def is_available(self) -> bool:
        if self.is_available_override is not None:
            return self.is_available_override
        return True

    def validate_configuration(self) -> None:
        if self.simulate_error is not None:
            raise self.simulate_error

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a deterministic response without making any network calls.
        """
        self.call_history.append(request)

        if self.simulate_error is not None:
            raise self.simulate_error

        if not request.messages:
            raise LLMResponseError("Request must contain at least one message.", provider_name=self.provider_name)

        last_msg = request.messages[-1]
        
        if self.canned_response is not None:
            text = self.canned_response
        else:
            text = (
                f"[Mock Response from {self.model_name}]: "
                f"Successfully processed prompt with {len(request.messages)} message(s). "
                f"Input preview: '{last_msg.content[:40]}...'"
            )

        finish_reason = "stop"
        effective_max_tokens = request.generation_config.max_output_tokens or self._config.max_output_tokens
        if effective_max_tokens is not None:
            words = text.split()
            if len(words) > effective_max_tokens:
                text = " ".join(words[:effective_max_tokens])
                finish_reason = "length"

        # Approximate token usage deterministically for testing accounting
        prompt_words = sum(len(m.content.split()) for m in request.messages)
        completion_words = len(text.split())
        
        usage = LLMUsageMetadata(
            prompt_tokens=prompt_words,
            completion_tokens=completion_words,
            total_tokens=prompt_words + completion_words,
        )

        return LLMResponse(
            content=text,
            provider_name=self.provider_name,
            model_name=self.model_name,
            usage=usage,
            finish_reason=finish_reason,
            raw_metadata={"mock": True, "messages_count": len(request.messages)},
        )

