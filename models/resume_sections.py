"""
Data models for resume text sections and structured parsing results.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class SectionType(str, Enum):
    """Canonical classification for resume sections."""
    CONTACT_INFORMATION = "Contact Information"
    PROFESSIONAL_SUMMARY = "Professional Summary"
    CAREER_OBJECTIVE = "Career Objective"
    TECHNICAL_SKILLS = "Technical Skills"
    WORK_EXPERIENCE = "Work Experience"
    INTERNSHIPS = "Internships"
    EDUCATION = "Education"
    PROJECTS = "Projects"
    CERTIFICATIONS = "Certifications"
    ACHIEVEMENTS = "Achievements"
    PUBLICATIONS = "Publications"
    LANGUAGES = "Languages"
    OTHER_INFORMATION = "Other Information"
    UNKNOWN = "Unknown"


class ResumeSection(BaseModel):
    """Represents a discrete detected section of a resume."""
    model_config = ConfigDict(frozen=True)

    section_type: SectionType = Field(
        ...,
        description="Canonical type classification for the section.",
    )
    original_heading: Optional[str] = Field(
        default=None,
        description="Verbatim heading text found in the document, if any.",
    )
    content: str = Field(
        ...,
        description="Text content contained within the section.",
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="1-based starting line number in the cleaned text.",
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="1-based ending line number in the cleaned text.",
    )


class ParsedResumeSections(BaseModel):
    """Complete structured breakdown of cleaned resume text and detected sections."""
    model_config = ConfigDict(frozen=True)

    original_text: str = Field(
        ...,
        description="The un-modified raw extracted text from the PDF.",
    )
    cleaned_text: str = Field(
        ...,
        description="Normalized and sanitized resume text.",
    )
    sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Ordered list of parsed resume sections.",
    )
    detected_section_names: list[str] = Field(
        default_factory=list,
        description="Names of canonical sections successfully identified in the document.",
    )
    undetected_section_names: list[str] = Field(
        default_factory=list,
        description="Names of canonical sections not detected in the document.",
    )

