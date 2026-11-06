"""
Data models for extracted resume entities: Skills, Work Experience, Education, and Projects.
Includes evidence tracking and ambiguity classification.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class SkillCategory(str, Enum):
    """Categorization for extracted technical skills."""
    PROGRAMMING_LANGUAGES = "Programming Languages"
    WEB_DEVELOPMENT = "Web Development"
    DATABASES = "Databases"
    CLOUD_COMPUTING = "Cloud Computing"
    DEVOPS_AND_INFRASTRUCTURE = "DevOps and Infrastructure"
    MACHINE_LEARNING = "Machine Learning"
    DATA_SCIENCE = "Data Science"
    DATA_ENGINEERING_AND_STREAMING = "Data Engineering and Streaming"
    TOOLS_AND_FRAMEWORKS = "Tools and Frameworks"
    OTHER_TECHNICAL_SKILLS = "Other Technical Skills"

    # Backward compatibility alias
    DEVOPS = "DevOps and Infrastructure"


class ExtractedSkill(BaseModel):
    """Represents a validated technical skill extracted from the resume."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="Standardized canonical name of the skill.",
        min_length=1,
    )
    original_mention: str = Field(
        ...,
        description="Verbatim mention of the skill in the document.",
        min_length=1,
    )
    category: SkillCategory = Field(
        default=SkillCategory.OTHER_TECHNICAL_SKILLS,
        description="Technical category the skill belongs to.",
    )
    source_section: str = Field(
        default="Technical Skills",
        description="Name of the resume section where this skill was identified.",
    )
    is_explicitly_listed: bool = Field(
        default=True,
        description="True if explicitly listed in skills section, False if mentioned in experience/projects.",
    )
    raw_evidence: Optional[str] = Field(
        default=None,
        description="Line or sentence context where the skill was identified.",
    )
    confidence: str = Field(
        default="CONFIRMED",
        description="'CONFIRMED' for direct taxonomy/explicit matches, 'AMBIGUOUS' for uncertain extractions.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if extraction carries lexical or contextual ambiguity.",
    )


class WorkExperience(BaseModel):
    """Represents an employment or internship history entry."""
    model_config = ConfigDict(frozen=True)

    job_title: Optional[str] = Field(
        default=None,
        description="Extracted job title or role.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Name of the employer or organization.",
    )
    employment_start: Optional[str] = Field(
        default=None,
        description="Start date or year of employment.",
    )
    employment_end: Optional[str] = Field(
        default=None,
        description="End date, year, or 'Present' if current.",
    )
    employment_type: Optional[str] = Field(
        default=None,
        description="Employment arrangement (e.g. Full-time, Internship, Contract).",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key bullet points or responsibilities performed.",
    )
    technologies: list[str] = Field(
        default_factory=list,
        description="Technologies explicitly mentioned in connection with this role.",
    )
    original_text: str = Field(
        ...,
        description="Verbatim text block representing this job entry.",
    )
    confidence: str = Field(
        default="CONFIRMED",
        description="'CONFIRMED' if title and company are resolved; 'AMBIGUOUS' if missing key fields.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if critical fields like company or title could not be confirmed.",
    )


class EducationRecord(BaseModel):
    """Represents an academic degree, certificate, or university qualification."""
    model_config = ConfigDict(frozen=True)

    degree: Optional[str] = Field(
        default=None,
        description="Degree awarded or pursued (e.g., Bachelor of Science, M.S., Ph.D.).",
    )
    field_of_study: Optional[str] = Field(
        default=None,
        description="Major, field of study, or specialization (e.g., Computer Science).",
    )
    institution: Optional[str] = Field(
        default=None,
        description="Name of the university, college, or academic institution.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date or year if available.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Graduation date, year, or expected graduation.",
    )
    grade_or_cgpa: Optional[str] = Field(
        default=None,
        description="GPA, CGPA, percentage, or honors distinction if explicitly listed.",
    )
    original_text: str = Field(
        ...,
        description="Verbatim text block representing this academic qualification.",
    )
    confidence: str = Field(
        default="CONFIRMED",
        description="'CONFIRMED' if degree and institution are resolved; 'AMBIGUOUS' if missing.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if degree or institution is missing or unverified.",
    )


class ProjectRecord(BaseModel):
    """Represents an engineering, personal, or academic project."""
    model_config = ConfigDict(frozen=True)

    project_name: str = Field(
        ...,
        description="Title or identifier of the project.",
        min_length=1,
    )
    description: str = Field(
        default="",
        description="Narrative overview and key accomplishments of the project.",
    )
    technologies: list[str] = Field(
        default_factory=list,
        description="Tools, libraries, and languages used in the project.",
    )
    project_url: Optional[str] = Field(
        default=None,
        description="URL link to project repository or live deployment, if available.",
    )
    original_text: str = Field(
        ...,
        description="Verbatim text block representing this project.",
    )
    confidence: str = Field(
        default="CONFIRMED",
        description="'CONFIRMED' if project name and content are well-defined; 'AMBIGUOUS' otherwise.",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if project boundaries or technologies are uncertain.",
    )


class ResumeEntities(BaseModel):
    """Top-level container for all structured resume entities."""
    model_config = ConfigDict(frozen=True)

    skills: list[ExtractedSkill] = Field(
        default_factory=list,
        description="Collection of all extracted skills across categories.",
    )
    experience: list[WorkExperience] = Field(
        default_factory=list,
        description="Chronological collection of work experience records.",
    )
    education: list[EducationRecord] = Field(
        default_factory=list,
        description="Collection of academic education and qualifications.",
    )
    projects: list[ProjectRecord] = Field(
        default_factory=list,
        description="Collection of highlighted engineering projects.",
    )
