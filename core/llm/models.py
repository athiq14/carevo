"""
Provider-independent Pydantic data models for LLM requests, responses, and configurations.
Ensures strict validation, consistent type hints, and confidential token handling.
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class MessageRole(str, Enum):
    """Canonical role classification for conversational messages."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """Individual conversational message unit."""
    model_config = ConfigDict(frozen=True)

    role: MessageRole = Field(
        ...,
        description="Originator role of the message: system, user, or assistant.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Text content of the message.",
    )


class LLMGenerationConfig(BaseModel):
    """Runtime hyperparameter parameters controlling text generation behavior."""
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; lower values are more deterministic.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens allowed in generation response. None indicates provider default.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability cutoff.",
    )
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="List of textual token patterns where generation should cease.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Request timeout in seconds.",
    )
    response_format: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional provider-specific response format specification (e.g. {'type': 'json_object'}).",
    )


class LLMUsageMetadata(BaseModel):
    """Token consumption accounting for LLM execution."""
    model_config = ConfigDict(frozen=True)

    prompt_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of prompt tokens evaluated, or None if provider does not report.",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of tokens generated in response, or None if unavailable.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total tokens consumed, or None if unavailable.",
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of internal reasoning/thinking tokens consumed, if reported by provider.",
    )


class LLMRequest(BaseModel):
    """Normalized payload delivered to an LLM provider."""
    model_config = ConfigDict(frozen=True)

    messages: list[LLMMessage] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of conversation messages.",
    )
    generation_config: LLMGenerationConfig = Field(
        default_factory=LLMGenerationConfig,
        description="Generation hyperparameters for this request.",
    )
    allow_fallback: bool = Field(
        default=True,
        description="Whether fallback to secondary provider is allowed if primary fails with transient error.",
    )
    is_sensitive: bool = Field(
        default=False,
        description="Whether the request payload contains sensitive candidate PII or confidential resume data.",
    )
    allow_cross_provider_fallback: bool = Field(
        default=False,
        description="Explicit opt-in permitting cross-provider fallback for sensitive candidate PII requests.",
    )

    def is_fallback_permitted(self) -> bool:
        """
        Determine whether cross-provider fallback is legally and architecturally permitted
        for this request.

        Rules:
        1. If fallback is globally disabled on the request (allow_fallback=False) -> False.
        2. If the request contains sensitive candidate data (is_sensitive=True), fallback
           across providers is strictly DISABLED unless explicitly permitted (allow_cross_provider_fallback=True).
        3. Otherwise -> True.
        """
        if not self.allow_fallback:
            return False
        if self.is_sensitive and not self.allow_cross_provider_fallback:
            return False
        return True

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        allow_fallback: bool = True,
        is_sensitive: bool = False,
        allow_cross_provider_fallback: bool = False,
    ) -> "LLMRequest":
        """Convenience factory for constructing standard single-turn prompt requests."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt text must contain non-whitespace characters.")

        msgs: list[LLMMessage] = []
        if system_prompt and system_prompt.strip():
            msgs.append(LLMMessage(role=MessageRole.SYSTEM, content=system_prompt.strip()))
        msgs.append(LLMMessage(role=MessageRole.USER, content=prompt.strip()))

        if generation_config is None:
            config_kwargs: dict[str, Any] = {}
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                config_kwargs["max_output_tokens"] = max_output_tokens
            gen_cfg = LLMGenerationConfig(**config_kwargs)
        else:
            updates: dict[str, Any] = {}
            if temperature is not None:
                updates["temperature"] = temperature
            if max_output_tokens is not None:
                updates["max_output_tokens"] = max_output_tokens
            if updates:
                gen_cfg = generation_config.model_copy(update=updates)
            else:
                gen_cfg = generation_config

        return cls(
            messages=msgs,
            generation_config=gen_cfg,
            allow_fallback=allow_fallback,
            is_sensitive=is_sensitive,
            allow_cross_provider_fallback=allow_cross_provider_fallback,
        )


class LLMResponse(BaseModel):
    """Normalized, provider-agnostic response generated by an LLM."""
    model_config = ConfigDict(frozen=True)

    content: str = Field(
        ...,
        description="Generated completion text from the provider.",
    )
    provider_name: str = Field(
        ...,
        min_length=1,
        description="Identifier of the provider executing the request (e.g., 'mock', 'gemini', 'groq').",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Canonical model identifier utilized for generation.",
    )
    usage: Optional[LLMUsageMetadata] = Field(
        default=None,
        description="Token accounting metadata, or None if unavailable from provider.",
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Termination justification returned by provider (e.g., 'stop', 'length').",
    )
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific supplementary metadata retained for debugging.",
    )


class LLMProviderConfig(BaseModel):
    """Provider configuration settings with secure credential handling."""
    model_config = ConfigDict(frozen=True)

    provider_name: str = Field(
        ...,
        min_length=1,
        description="Name of the LLM provider service (e.g. 'gemini', 'groq', 'mock').",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Target model identifier (e.g. 'gemini-3.8-flash', 'openai/gpt-oss-120b').",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Private API key for provider access. Never logged or serialized in cleartext.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature.",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Default maximum tokens for generation.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="Default network timeout in seconds.",
    )

    @property
    def is_api_key_configured(self) -> bool:
        """Indicate whether a non-empty API key is present."""
        return bool(self.api_key and self.api_key.strip())

    @field_serializer("api_key", when_used="always")
    def serialize_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Prevent plain-text secret leakage during model serialization."""
        if api_key and api_key.strip():
            return "[CONFIGURED]"
        return None

    def __repr__(self) -> str:
        """Protect sensitive credentials from leaking into debug representations."""
        masked_key = "[CONFIGURED]" if self.is_api_key_configured else "[NOT_CONFIGURED]"
        return (
            f"LLMProviderConfig(provider_name='{self.provider_name}', "
            f"model_name='{self.model_name}', "
            f"api_key={masked_key}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"timeout={self.timeout})"
        )

    def __str__(self) -> str:
        """Protect sensitive credentials from leaking into string conversions."""
        return self.__repr__()

