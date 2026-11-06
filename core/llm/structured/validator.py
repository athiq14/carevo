"""
Pydantic schema validation engine for structured LLM outputs.
Validates arbitrary BaseModel schemas, nested objects, enums, lists, and optionals.
Produces sanitized validation diagnostics without leaking candidate PII.
"""
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from core.llm.exceptions import redact_secrets
from core.llm.structured.exceptions import StructuredValidationError

T = TypeVar("T", bound=BaseModel)


def format_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format and sanitize Pydantic validation error objects.
    Redacts any raw input values from the error diagnostics to protect candidate PII.
    """
    sanitized: list[dict[str, Any]] = []
    for err in errors:
        loc_parts = err.get("loc", ())
        loc_str = " -> ".join(str(p) for p in loc_parts) if loc_parts else "[root]"
        msg = err.get("msg", "Invalid value")
        err_type = err.get("type", "value_error")

        sanitized.append({
            "field": loc_str,
            "message": redact_secrets(msg),
            "type": err_type,
        })
    return sanitized


def validate_against_schema(
    data: Any,
    schema_class: Type[T],
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    raw_content: Optional[str] = None,
    attempts: int = 1,
) -> T:
    """
    Validate parsed data dictionary against a target Pydantic BaseModel schema.

    Args:
        data: Parsed dictionary or structure.
        schema_class: Target Pydantic model class.
        provider_name: Canonical provider name for error tracing.
        model_name: Model name for error tracing.
        raw_content: Raw LLM output for debugging.
        attempts: Attempt count.

    Returns:
        Validated Pydantic instance of type T.

    Raises:
        StructuredValidationError: If validation fails on missing fields, bad types, enums, etc.
    """
    if not issubclass(schema_class, BaseModel):
        raise TypeError(f"schema_class must be a subclass of pydantic.BaseModel, got {schema_class}")

    try:
        validated_instance = schema_class.model_validate(data)
        return validated_instance
    except ValidationError as exc:
        sanitized_errors = format_validation_errors(exc.errors())
        error_details = "; ".join(
            f"Field '{e['field']}': {e['message']} ({e['type']})"
            for e in sanitized_errors
        )
        raise StructuredValidationError(
            message=f"Schema validation failed against '{schema_class.__name__}': {error_details}",
            validation_errors=sanitized_errors,
            provider_name=provider_name,
            model_name=model_name,
            raw_content=raw_content,
            attempts=attempts,
        ) from exc
    except Exception as exc:
        raise StructuredValidationError(
            message=f"Unexpected error validating against '{schema_class.__name__}': {redact_secrets(str(exc))}",
            provider_name=provider_name,
            model_name=model_name,
            raw_content=raw_content,
            attempts=attempts,
        ) from exc

