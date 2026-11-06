"""
Domain models and data contracts for intelligent job description matching.
Defines immutable Pydantic V2 schemas for job requirements, parsed profiles,
skill matches, gaps, multi-dimensional breakdowns, project evidence, and match results.
Enforces strict score boundaries, weight normalization, and evidence semantics.
"""
from enum import Enum
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =====================================================================
# Enumerations
# =====================================================================

class RequirementCategory(str, Enum):
    """Categorization of job requirements extracted from job descriptions."""
    TECHNICAL_SKILL = "TECHNICAL_SKILL"
    EXPERIENCE_TENURE = "EXPERIENCE_TENURE"
    ROLE_RELEVANCE = "ROLE_RELEVANCE"
    EDUCATION = "EDUCATION"
    CERTIFICATION = "CERTIFICATION"
    PROJECT = "PROJECT"
    RESPONSIBILITY = "RESPONSIBILITY"


class RequirementPriority(str, Enum):
    """Priority weighting assigned to a job requirement."""
    REQUIRED = "REQUIRED"      # Mandatory must-have qualification
    PREFERRED = "PREFERRED"    # Desired / nice-to-have qualification
    OPTIONAL = "OPTIONAL"      # Secondary bonus qualification


class EvidenceStatus(str, Enum):
    """
    Evidence classification status reflecting resume examination results.
    Preserves the distinction between missing resume evidence and candidate ability.
    """
    VERIFIED_EVIDENCE = "VERIFIED_EVIDENCE"   # Direct or recognized synonym evidence found
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"     # Mapped related technology or partial tenure
    MISSING_EVIDENCE = "MISSING_EVIDENCE"     # Searched section; requirement absent in resume
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"     # Qualitative / unstated; insufficient info to assess


class SkillMatchType(str, Enum):
    """Precision tier for technical skill matching against the taxonomy."""
    EXACT_MATCH = "EXACT_MATCH"                 # Verbatim token or canonical name match
    SYNONYM_MATCH = "SYNONYM_MATCH"             # Verified interchangeable standard alias
    RELATED_TECHNOLOGY = "RELATED_TECHNOLOGY"   # Pre-mapped architectural neighbor (e.g. Flask -> FastAPI)
    NO_MATCH = "NO_MATCH"                       # Unrelated or unevidenced skill
    UNKNOWN = "UNKNOWN"                         # Insufficient context to classify


class ExperienceTier(str, Enum):
    """Discrete role relevance classification tiers."""
    DIRECT_ALIGNMENT = "DIRECT_ALIGNMENT"           # 100% - Matching title & core duties
    ADJACENT_ALIGNMENT = "ADJACENT_ALIGNMENT"       # 75% - Adjacent technical domain or junior title
    RELEVANT_INTERNSHIP = "RELEVANT_INTERNSHIP"     # 60% standard / 80% entry-level
    APPLIED_PROJECTS = "APPLIED_PROJECTS"           # 40% - Substantial applied project evidence
    UNRELATED_BACKGROUND = "UNRELATED_BACKGROUND"   # 0% - Non-technical employment
    UNKNOWN = "UNKNOWN"                             # Neutralized - no career records present


class EvaluationStatus(str, Enum):
    """Lifecycle assessment validity status."""
    FULL_EVALUATION = "FULL_EVALUATION"                   # Meets minimum thresholds (>=3 skills, >=4 reqs)
    PROVISIONAL_EVALUATION = "PROVISIONAL_EVALUATION"     # Sparse JD; below minimum threshold
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"       # Cannot compute score; missing core criteria


class EvidenceVerificationLevel(str, Enum):
    """
    Verification certainty level for candidate evidence.
    Strictly distinguishes candidate self-reported claims from verified artifacts.
    """
    SELF_REPORTED = "SELF_REPORTED"                     # Candidate-written text in resume
    CANDIDATE_PROVIDED_URL = "CANDIDATE_PROVIDED_URL"   # URL provided by candidate but invalid or unaccepted host
    STRUCTURALLY_VALID_URL = "STRUCTURALLY_VALID_URL"   # Syntactically valid HTTPS URL to accepted artifact/portfolio host
    ARTIFACT_SUPPORTED = "ARTIFACT_SUPPORTED"           # Verified artifact content inspected
    INDEPENDENTLY_VERIFIED = "INDEPENDENTLY_VERIFIED"   # Third-party authenticated credential/assessment


class CandidateFitLevel(str, Enum):
    """High-level candidate alignment classification."""
    EXCEPTIONAL = "EXCEPTIONAL"   # 90.0% - 100.0%
    STRONG = "STRONG"             # 75.0% - 89.9%
    MODERATE = "MODERATE"         # 60.0% - 74.9%
    DEVELOPING = "DEVELOPING"     # 40.0% - 59.9%
    LOW = "LOW"                   # < 40.0%
    PROVISIONAL = "PROVISIONAL"   # Sparse JD; evaluation is non-final regardless of score


# =====================================================================
# Domain Models
# =====================================================================

class JobRequirement(BaseModel):
    """A discrete, structured qualification requirement extracted from a job description."""
    model_config = ConfigDict(frozen=True)

    requirement_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the requirement (e.g. slug or UUID).",
    )
    requirement_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim excerpt or clause from the job description.",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        description="Standardized name (e.g. canonical skill or qualification concept).",
    )
    category: RequirementCategory = Field(
        ...,
        description="Structural category of the requirement.",
    )
    priority: RequirementPriority = Field(
        default=RequirementPriority.REQUIRED,
        description="Priority weighting (REQUIRED, PREFERRED, OPTIONAL).",
    )
    evidence_expectation: Optional[str] = Field(
        default=None,
        description="Expectation for evidence matching (e.g., 'minimum 3 years', 'bachelor degree').",
    )
    source_section: Optional[str] = Field(
        default=None,
        description="Section of the JD where this requirement appeared (e.g. 'Requirements').",
    )

    @field_validator("requirement_id", "requirement_text", "canonical_name")
    @classmethod
    def _strip_and_validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field must contain non-whitespace characters.")
        return s


class JobDescriptionProfile(BaseModel):
    """Consolidated, structured profile extracted and normalized from raw job description text."""
    model_config = ConfigDict(frozen=True)

    job_title: Optional[str] = Field(
        default=None,
        description="Target job title or role name.",
    )
    company: Optional[str] = Field(
        default=None,
        description="Hiring company or organization name, if provided.",
    )
    original_description: str = Field(
        ...,
        min_length=1,
        description="Verbatim, unmodified text of the job description.",
    )
    normalized_requirements: list[JobRequirement] = Field(
        default_factory=list,
        description="Ordered sequence of discrete, deduplicated requirements.",
    )
    experience_requirement: Optional[str] = Field(
        default=None,
        description="Summary of minimum professional experience requirements.",
    )
    education_requirement: Optional[str] = Field(
        default=None,
        description="Summary of academic education and degree expectations.",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key role duties and responsibilities extracted from the JD.",
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="List of mandatory technical skill names.",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="List of preferred / nice-to-have technical skill names.",
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Professional certifications requested by the JD.",
    )
    project_requirements: list[str] = Field(
        default_factory=list,
        description="Explicit project, repository, or portfolio expectations requested by the JD.",
    )
    requirement_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Deterministic requirement count metrics (skills, experience, education, etc.).",
    )
    evaluation_eligibility: EvaluationStatus = Field(
        default=EvaluationStatus.FULL_EVALUATION,
        description="Assessment eligibility: FULL_EVALUATION or PROVISIONAL_EVALUATION based on criteria sufficiency.",
    )
    processing_warnings: list[str] = Field(
        default_factory=list,
        description="Notices or warnings encountered during parsing (e.g. sparse content).",
    )

    @property
    def is_provisional(self) -> bool:
        """Indicates whether this job description criteria is sparse, requiring provisional match evaluation."""
        return self.evaluation_eligibility == EvaluationStatus.PROVISIONAL_EVALUATION

    @property
    def total_requirement_count(self) -> int:
        """Total assessable deduplicated requirements count."""
        return self.requirement_counts.get("total_requirements", len(self.normalized_requirements))

    @property
    def technical_skill_count(self) -> int:
        """Total technical skill requirements count (required + preferred)."""
        return self.requirement_counts.get("technical_skills", len(self.required_skills) + len(self.preferred_skills))

    @field_validator("original_description")
    @classmethod
    def _validate_original_description(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("original_description cannot be empty or purely whitespace.")
        return s


class SkillMatch(BaseModel):
    """Detailed matching record for an individual technical skill requirement."""
    model_config = ConfigDict(frozen=True)

    skill_name: str = Field(
        ...,
        min_length=1,
        description="Skill name as requested in the job description.",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        description="Standardized canonical taxonomy name.",
    )
    requirement_priority: RequirementPriority = Field(
        ...,
        description="Whether this skill is REQUIRED, PREFERRED, or OPTIONAL.",
    )
    match_type: SkillMatchType = Field(
        ...,
        description="Matching tier: EXACT_MATCH, SYNONYM_MATCH, RELATED_TECHNOLOGY, NO_MATCH, or UNKNOWN.",
    )
    evidence_status: EvidenceStatus = Field(
        ...,
        description="Evidence status: VERIFIED_EVIDENCE, PARTIAL_EVIDENCE, MISSING_EVIDENCE, UNKNOWN_EVIDENCE.",
    )
    resume_evidence_references: list[str] = Field(
        default_factory=list,
        description="Verbatim resume snippets, role titles, or section references where evidence was found.",
    )
    jd_evidence_references: list[str] = Field(
        default_factory=list,
        description="Verbatim JD excerpt stating the skill requirement.",
    )
    match_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional probabilistic confidence score between 0.0 and 1.0.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable justification for the match classification.",
    )


class SkillGap(BaseModel):
    """Identified technical skill deficiency with actionable learning recommendation."""
    model_config = ConfigDict(frozen=True)

    skill_name: str = Field(
        ...,
        min_length=1,
        description="Name of the skill missing or partially evidenced in the resume.",
    )
    priority: RequirementPriority = Field(
        ...,
        description="Importance of the missing skill for the target role.",
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.MISSING_EVIDENCE,
        description="Evidence status (MISSING_EVIDENCE or PARTIAL_EVIDENCE).",
    )
    gap_explanation: str = Field(
        ...,
        min_length=1,
        description="Technical explanation detailing why this is flagged as a gap.",
    )
    suggested_learning_direction: Optional[str] = Field(
        default=None,
        description="Actionable upskilling or learning path to bridge the gap.",
    )


class MatchBreakdown(BaseModel):
    """Evaluation score, weights, and status for an individual assessment dimension."""
    model_config = ConfigDict(frozen=True)

    dimension_name: str = Field(
        ...,
        min_length=1,
        description="Name of the evaluation dimension (e.g. 'Technical Skills', 'Experience').",
    )
    baseline_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Unadjusted baseline weight assigned to this dimension (e.g. 0.45).",
    )
    active_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized active weight after redistribution (max 0.60 when capped).",
    )
    dimension_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Calculated score for this dimension strictly bounded [0.0, 100.0].",
    )
    evidence_status: EvidenceStatus = Field(
        ...,
        description="Evidence status reflecting this dimension's data sufficiency.",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this dimension is active in scoring or neutralized.",
    )
    neutralization_explanation: Optional[str] = Field(
        default=None,
        description="Documented rationale if the dimension was neutralized (active_weight = 0.0).",
    )

    @model_validator(mode="after")
    def _validate_active_state_consistency(self) -> "MatchBreakdown":
        if not self.is_active and self.active_weight > 0.0:
            raise ValueError("A neutralized dimension (is_active=False) must have active_weight=0.0.")
        return self


# Backward-compatibility alias
DimensionBreakdown = MatchBreakdown


class ProjectEvidence(BaseModel):
    """Audit record evaluating candidate project implementation against role requirements."""
    model_config = ConfigDict(frozen=True)

    project_identifier: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the project record.",
    )
    project_name: str = Field(
        ...,
        min_length=1,
        description="Title or name of the project.",
    )
    relevant_technologies: list[str] = Field(
        default_factory=list,
        description="JD-relevant tools verified as implemented in this project.",
    )
    technology_implementation_evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim description clauses establishing active engineering implementation.",
    )
    artifact_url: Optional[str] = Field(
        default=None,
        description="Public repository or deployment URL demonstrating code existence.",
    )
    evidence_verification_level: EvidenceVerificationLevel = Field(
        default=EvidenceVerificationLevel.SELF_REPORTED,
        description="Verification level: SELF_REPORTED, ARTIFACT_SUPPORTED, or INDEPENDENTLY_VERIFIED.",
    )
    self_reported_metrics: list[str] = Field(
        default_factory=list,
        description="Candidate-described performance or throughput metrics (self-reported claims).",
    )
    toolchain_interaction_description: Optional[str] = Field(
        default=None,
        description="Narrative describing how >=3 tools interact in an operational workflow.",
    )
    responsibility_alignment_evidence: list[str] = Field(
        default_factory=list,
        description="Clauses matching target role responsibilities.",
    )
    calculated_pillar_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Diagnostic breakdown across project pillars (tech_coverage, context_depth, duty_align).",
    )

    @field_validator("artifact_url")
    @classmethod
    def _validate_artifact_url_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            s = v.strip()
            if not s:
                return None
            return s
        return None


class MatchExplanation(BaseModel):
    """
    Strict, structured, evidence-grounded career matching explanation generated by the LLM layer.
    Purely qualitative and explanatory; strictly immutable with respect to deterministic scoring.
    """
    model_config = ConfigDict(frozen=True)

    overall_match_summary: str = Field(
        ...,
        description="Comprehensive narrative summary synthesizing the candidate's alignment with the role.",
    )
    key_strengths: list[str] = Field(
        ...,
        description="Distinct verified candidate qualifications and evidence directly supporting fit.",
    )
    skill_gaps_assessment: list[str] = Field(
        ...,
        description="Objective breakdown of missing or partially evidenced skills without disparagement.",
    )
    experience_assessment: str = Field(
        ...,
        description="Detailed review of professional employment tenure, title relevance, and responsibility overlap.",
    )
    education_assessment: str = Field(
        ...,
        description="Evaluation of academic qualification level and discipline alignment.",
    )
    project_assessment: str = Field(
        ...,
        description="Contextual appraisal of engineering projects, tooling interaction, and self-reported metrics.",
    )
    improvement_recommendations: list[str] = Field(
        ...,
        description="Actionable, practical technical recommendations for closing identified gaps.",
    )
    interview_preparation_suggestions: list[str] = Field(
        ...,
        description="Specific technical topics, behavioral scenarios, or portfolio talking points for interview readiness.",
    )
    evidence_limitations_and_warnings: list[str] = Field(
        ...,
        description="Explicit caveats regarding missing JD specifications, uninspected repositories, or self-reported metrics.",
    )


class JobMatchResult(BaseModel):
    """
    Consolidated, Pydantic-validated result of intelligent job description matching.
    Includes deterministic scores, dimensional breakdown, evidence classifications,
    project audits, and optional qualitative AI explanations.
    """
    model_config = ConfigDict(frozen=True)

    job_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata describing target position, company, and submission parameters.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Deterministic overall compatibility score bounded [0.0, 100.0].",
    )
    evaluation_status: EvaluationStatus = Field(
        ...,
        description="Assessment status: FULL_EVALUATION, PROVISIONAL_EVALUATION, or INSUFFICIENT_EVIDENCE.",
    )
    fit_classification: CandidateFitLevel = Field(
        ...,
        description="Overall fit classification (EXCEPTIONAL, STRONG, MODERATE, DEVELOPING, LOW, PROVISIONAL).",
    )
    active_evaluation_coverage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of standard role criteria assessable from the JD (max 100.0, or 60.0 for 1-dimension).",
    )
    unassessed_reserve: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of criteria unstated by employer held in reserve (e.g. 40.0 for 1-dimension).",
    )
    match_breakdown: list[MatchBreakdown] = Field(
        default_factory=list,
        description="Detailed scoring and weight redistribution across evaluation dimensions.",
    )
    matched_skills: list[SkillMatch] = Field(
        default_factory=list,
        description="List of skills verified through exact or synonym matches.",
    )
    partial_matches: list[SkillMatch] = Field(
        default_factory=list,
        description="List of skills partially evidenced through pre-mapped related technologies.",
    )
    missing_evidence: list[SkillGap] = Field(
        default_factory=list,
        description="Requirements with no supporting evidence found in the resume.",
    )
    unknown_evidence: list[SkillMatch] = Field(
        default_factory=list,
        description="Requirements where resume information is insufficient to make a determination.",
    )
    project_evidence: list[ProjectEvidence] = Field(
        default_factory=list,
        description="Audit records for candidate project portfolio.",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Highlighted candidate competencies matching role requirements.",
    )
    skill_gaps: list[SkillGap] = Field(
        default_factory=list,
        description="Identified candidate skill gaps with suggested learning paths.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Strategic recommendations for closing gaps and optimizing alignment.",
    )
    processing_warnings: list[str] = Field(
        default_factory=list,
        description="System warnings or diagnostics (e.g. sparse JD notice, unstated criteria).",
    )
    llm_explanation_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provenance metadata for qualitative AI explanations (provider, model, token usage).",
    )
    ai_explanation: Optional[MatchExplanation] = Field(
        default=None,
        description="Structured qualitative explanation generated by the LLM layer, if requested.",
    )

    @model_validator(mode="after")
    def _validate_provisional_consistency(self) -> "JobMatchResult":
        """Ensure provisional evaluations are clearly reflected in fit classification."""
        if self.evaluation_status == EvaluationStatus.PROVISIONAL_EVALUATION:
            if self.fit_classification != CandidateFitLevel.PROVISIONAL:
                # Enforce that provisional status overrides numerical fit label
                object.__setattr__(self, "fit_classification", CandidateFitLevel.PROVISIONAL)
        return self

    def get_dimension(self, dimension_name: str) -> Optional[MatchBreakdown]:
        """Helper to look up a dimension breakdown by case-insensitive name."""
        name_lower = dimension_name.strip().lower()
        for d in self.match_breakdown:
            if d.dimension_name.strip().lower() == name_lower:
                return d
        return None

