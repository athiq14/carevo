"""
Structured LLM outputs and schema validation module.
Provides provider-independent parsing, Pydantic validation, and bounded correction retries.
"""
from core.llm.structured.engine import StructuredOutputEngine
from core.llm.structured.exceptions import (
    CorrectionExhaustedError,
    JSONParsingError,
    StructuredOutputError,
    StructuredValidationError,
)
from core.llm.structured.models import (
    StructuredOutputRequest,
    StructuredOutputResponse,
)
from core.llm.structured.parser import (
    extract_json_text,
    parse_json_payload,
)
from core.llm.structured.prompt_builder import (
    build_correction_instruction,
    build_schema_instruction,
)
from core.llm.structured.validator import (
    format_validation_errors,
    validate_against_schema,
)

__all__ = [
    "CorrectionExhaustedError",
    "JSONParsingError",
    "StructuredOutputEngine",
    "StructuredOutputError",
    "StructuredOutputRequest",
    "StructuredOutputResponse",
    "StructuredValidationError",
    "build_correction_instruction",
    "build_schema_instruction",
    "extract_json_text",
    "format_validation_errors",
    "parse_json_payload",
    "validate_against_schema",
]

