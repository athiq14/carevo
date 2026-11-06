"""
Prompt builder utilities for structured output schema injection and correction prompting.
Extracts Pydantic model JSON schemas and builds provider-agnostic instructions.
"""
import json
from typing import Type
from pydantic import BaseModel


def build_schema_instruction(schema_class: Type[BaseModel]) -> str:
    """
    Generate structured output instruction embedding the target Pydantic JSON schema.
    """
    schema_dict = schema_class.model_json_schema()
    # Pretty-format JSON schema
    schema_json = json.dumps(schema_dict, indent=2)

    return (
        f"You must respond ONLY with a single valid JSON object that strictly adheres to the following JSON Schema:\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Strict Requirements:\n"
        f"1. Output ONLY the JSON object. Do not include introductory text, conversational pleasantries, or explanations.\n"
        f"2. Ensure all required fields, correct data types, and enum values match the schema exactly.\n"
        f"3. Do not invent or hallucinate missing data to satisfy requirements."
    )


def build_correction_instruction(
    schema_class: Type[BaseModel],
    error_summary: str,
) -> str:
    """
    Generate an explicit correction prompt detailing the validation error and target schema.
    """
    schema_dict = schema_class.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    return (
        f"Your previous response was rejected due to JSON or schema validation failures:\n"
        f"{error_summary}\n\n"
        f"Please correct the output and return ONLY a valid JSON object matching the schema:\n"
        f"```json\n{schema_json}\n```\n\n"
        f"Strict Requirements:\n"
        f"1. Fix every reported error.\n"
        f"2. Return ONLY the valid JSON object without commentary, explanations, or extraneous text."
    )

