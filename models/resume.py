"""
Resume data models and validation schemas.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class ExtractionStatus(str, Enum):
    """Status indicator for document extraction pipeline."""
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"


class ResumeExtractionResult(BaseModel):
    """
    Validated result of resume PDF text and metadata extraction.
    """
    model_config = ConfigDict(frozen=True)

    filename: str = Field(
        ...,
        description="Name of the processed resume file.",
        min_length=1,
    )
    page_count: int = Field(
        ...,
        description="Total page count of the document.",
        ge=0,
    )
    extracted_text: str = Field(
        ...,
        description="Clean concatenated plain text extracted across all pages.",
    )
    character_count: int = Field(
        ...,
        description="Total character count of the extracted text.",
        ge=0,
    )
    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.SUCCESS,
        description="Overall execution status of the extraction.",
    )
    requires_ocr: bool = Field(
        default=False,
        description="Indicates whether the document has insufficient text and likely requires OCR.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Optional diagnostic or error message if status is WARNING or FAILED.",
    )

