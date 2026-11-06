"""
Generic data models for structured output generation requests and responses.
Supports arbitrary Pydantic schema validation without hardcoding domain-specific models.
"""
from typing import Any, Generic, Optional, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field

from core.llm.models import LLMRequest, LLMResponse

T = TypeVar("T", bound=BaseModel)


class StructuredOutputRequest(BaseModel, Generic[T]):
    """Specification for structured output generation using a target Pydantic schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    request: LLMRequest = Field(
        ...,
        description="Underlying normalized LLM request containing conversation context.",
    )
    schema_class: Type[T] = Field(
        ...,
        description="Target Pydantic model class that the response must validate against.",
    )
    max_attempts: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Maximum total generation attempts (1 initial + at most 1 correction). Strict upper bound of 2.",
    )
    system_instruction: Optional[str] = Field(
        default=None,
        description="Optional additional system instructions specifically guiding structured generation.",
    )


class StructuredOutputResponse(BaseModel, Generic[T]):
    """Normalized container for validated structured outputs and diagnostic audit metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: T = Field(
        ...,
        description="Validated Pydantic object instance conforming to requested schema.",
    )
    raw_response: LLMResponse = Field(
        ...,
        description="Underlying provider-agnostic response payload for diagnostics.",
    )
    raw_text: str = Field(
        ...,
        description="Raw text string returned by the provider before parsing.",
    )
    attempts: int = Field(
        ...,
        ge=1,
        le=2,
        description="Total number of attempts executed (1 = initial success, 2 = recovered via correction).",
    )
    correction_occurred: bool = Field(
        default=False,
        description="Indicates whether an error correction retry was executed before successful validation.",
    )
    provider_name: str = Field(
        ...,
        min_length=1,
        description="Canonical identifier of provider that generated the response.",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Canonical identifier of model used for generation.",
    )
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic metadata, including usage, finish reason, and audit trace.",
    )

