"""
Unified Resume Profile data models and validation schemas.
Combines extraction metadata, normalized text representations, parsed section hierarchy,
and structured entity records into an immutable, consolidated profile schema.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from models.resume import ExtractionStatus, ResumeExtractionResult
from models.resume_sections import (
    ParsedResumeSections,
    ResumeSection,
    SectionType,
)
from models.resume_entities import (
    EducationRecord,
    ExtractedSkill,
    ProjectRecord,
    ResumeEntities,
    SkillCategory,
    WorkExperience,
)


class ProfileProcessingStatus(str, Enum):
    """Execution status for unified resume profile generation."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ResumeProfile(BaseModel):
    """
    Unified, validated resume profile combining extraction, cleaning,
    section breakdown, and structured entity extraction.
    Designed for consumption by downstream career intelligence modules.
    """
    model_config = ConfigDict(frozen=True)

    # Document extraction metadata
    filename: str = Field(
        ...,
        description="Name of the source resume file.",
        min_length=1,
    )
    page_count: int = Field(
        default=0,
        ge=0,
        description="Total page count of the processed document.",
    )
    raw_character_count: int = Field(
        default=0,
        ge=0,
        description="Total character count extracted from the PDF before cleaning.",
    )
    cleaned_character_count: int = Field(
        default=0,
        ge=0,
        description="Total character count after normalization and sanitization.",
    )
    requires_ocr: bool = Field(
        default=False,
        description="Indicates whether the document lacked extractable text and requires OCR.",
    )
    ocr_warning: Optional[str] = Field(
        default=None,
        description="Warning message regarding OCR necessity if applicable.",
    )

    # Processing lifecycle status
    status: ProfileProcessingStatus = Field(
        default=ProfileProcessingStatus.SUCCESS,
        description="Overall processing status: SUCCESS, PARTIAL (e.g. OCR needed), or FAILED.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Extraction warnings, notices, or ambiguity summaries.",
    )
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the unified profile was constructed.",
    )

    # Text representations
    original_text: str = Field(
        default="",
        description="Verbatim, unmodified text extracted directly from the PDF.",
    )
    cleaned_text: str = Field(
        default="",
        description="Sanitized, normalized resume text with normalized line endings and whitespace.",
    )

    # Section structure
    sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Ordered sequence of detected resume sections.",
    )
    detected_section_names: list[str] = Field(
        default_factory=list,
        description="Canonical section names identified in the document.",
    )
    undetected_section_names: list[str] = Field(
        default_factory=list,
        description="Canonical section names not detected in the document.",
    )

    # Structured entities
    skills: list[ExtractedSkill] = Field(
        default_factory=list,
        description="Standardized technical skills categorized across domains.",
    )
    experience: list[WorkExperience] = Field(
        default_factory=list,
        description="Chronological work and internship experience records.",
    )
    education: list[EducationRecord] = Field(
        default_factory=list,
        description="Academic degrees and qualifications.",
    )
    projects: list[ProjectRecord] = Field(
        default_factory=list,
        description="Highlighted technical and engineering projects.",
    )

    @property
    def has_ambiguities(self) -> bool:
        """Check whether any extracted entity is flagged as ambiguous or uncertain."""
        return (
            any(s.is_ambiguous for s in self.skills)
            or any(e.is_ambiguous for e in self.experience)
            or any(ed.is_ambiguous for ed in self.education)
            or any(p.is_ambiguous for p in self.projects)
        )

    def get_skills_by_category(self) -> dict[SkillCategory, list[ExtractedSkill]]:
        """Group extracted skills by their assigned canonical category."""
        grouped: dict[SkillCategory, list[ExtractedSkill]] = {}
        for skill in self.skills:
            grouped.setdefault(skill.category, []).append(skill)
        return grouped

    def get_section_by_type(self, section_type: SectionType) -> Optional[ResumeSection]:
        """Retrieve the first matching section for a given canonical SectionType."""
        for sec in self.sections:
            if sec.section_type == section_type:
                return sec
        return None

    @classmethod
    def from_components(
        cls,
        pdf_result: ResumeExtractionResult,
        cleaned_text: str,
        parsed_sections: ParsedResumeSections,
        entities: ResumeEntities,
        status: Optional[ProfileProcessingStatus] = None,
        warnings: Optional[list[str]] = None,
    ) -> "ResumeProfile":
        """
        Factory method to assemble a unified ResumeProfile from pipeline stage outputs.
        """
        warnings_list: list[str] = list(warnings) if warnings is not None else []
        ocr_warning: Optional[str] = None

        if pdf_result.requires_ocr:
            ocr_msg = (
                pdf_result.error_message
                or "Document appears to be a scanned image or lacks sufficient extractable text. OCR is recommended."
            )
            if ocr_msg not in warnings_list:
                warnings_list.append(ocr_msg)
            ocr_warning = ocr_msg

        if status is None:
            if pdf_result.extraction_status == ExtractionStatus.FAILED:
                derived_status = ProfileProcessingStatus.FAILED
            elif pdf_result.requires_ocr:
                derived_status = ProfileProcessingStatus.PARTIAL
            elif not parsed_sections.detected_section_names and len(cleaned_text.strip()) > 0:
                derived_status = ProfileProcessingStatus.PARTIAL
            else:
                derived_status = ProfileProcessingStatus.SUCCESS
        else:
            derived_status = status

        return cls(
            filename=pdf_result.filename,
            page_count=pdf_result.page_count,
            raw_character_count=pdf_result.character_count,
            cleaned_character_count=len(cleaned_text),
            requires_ocr=pdf_result.requires_ocr,
            ocr_warning=ocr_warning,
            status=derived_status,
            warnings=warnings_list,
            original_text=pdf_result.extracted_text,
            cleaned_text=cleaned_text,
            sections=parsed_sections.sections,
            detected_section_names=parsed_sections.detected_section_names,
            undetected_section_names=parsed_sections.undetected_section_names,
            skills=entities.skills,
            experience=entities.experience,
            education=entities.education,
            projects=entities.projects,
        )
