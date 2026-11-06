"""
Domain models and immutable data contracts for Phase 5 Agentic Workflows.
Defines immutable Pydantic V2 schemas for workflow deliverables, evidence provenance,
validation reports, telemetry events, and goal configurations.
Enforces strict bounds, field validations, and separation of candidate claims from recommendations.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    JobMatchResult,
    MatchBreakdown,
    ProjectEvidence,
    SkillGap,
    SkillMatch,
)


# =============================================================================
# Core Enums
# =============================================================================

class ProvenanceTier(str, Enum):
    """
    Classification of evidence entry mechanisms into the workflow system.
    None of these tiers imply independent third-party factual verification.
    """
    RESUME_EXTRACTED = "RESUME_EXTRACTED"   # Directly parsed from candidate PDF resume
    USER_CONFIRMED = "USER_CONFIRMED"       # Ambiguous resume item explicitly confirmed by user
    USER_SUPPLIED = "USER_SUPPLIED"         # New qualification or detail provided by user during HITL


class MetricDisclosureStatus(str, Enum):
    """Disclosure status for quantitative achievements or performance claims."""
    SELF_REPORTED_METRIC = "SELF_REPORTED_METRIC"   # Candidate self-reported quantitative narrative
    NOT_APPLICABLE = "NOT_APPLICABLE"


class URLVerificationStatus(str, Enum):
    """
    Offline syntax validation status for candidate-provided web links.
    Strictly decoupled from third-party code or repository content verification.
    """
    SYNTACTICALLY_VALID_HTTPS_URL = "SYNTACTICALLY_VALID_HTTPS_URL"  # Approved host, valid HTTPS syntax
    REJECTED_URL = "REJECTED_URL"                                      # Disallowed port, deceptive host, etc.
    NO_URL_PROVIDED = "NO_URL_PROVIDED"


class UserWorkflowGoal(str, Enum):
    """Workflow execution goal selected explicitly by the user."""
    FULL_PREPARATION = "FULL_PREPARATION"          # Remediation -> Resume -> Interview -> Audit
    GAP_REMEDIATION = "GAP_REMEDIATION"            # Remediation -> Audit
    RESUME_OPTIMIZATION = "RESUME_OPTIMIZATION"    # Resume -> Audit
    INTERVIEW_DRILL = "INTERVIEW_DRILL"            # Interview -> Audit


class ClaimContext(str, Enum):
    """
    Contextual scope for skill and qualification assertions.
    Differentiates candidate historical claims from forward-looking recommendations.
    """
    CANDIDATE_ASSERTION = "CANDIDATE_ASSERTION"          # Candidate claimed ability; requires strict whitelist
    ADVISORY_RECOMMENDATION = "ADVISORY_RECOMMENDATION"  # Forward-looking learning goal; may cite missing JD skills


class ValidationViolationType(str, Enum):
    """Granular violation taxonomy for deterministic validation."""
    MISSING_EVIDENCE_REFERENCE = "MISSING_EVIDENCE_REFERENCE"
    UNGROUNDED_SKILL_CLAIM = "UNGROUNDED_SKILL_CLAIM"
    TITLE_ALTERATION_CLAIM = "TITLE_ALTERATION_CLAIM"
    ORGANIZATION_ALTERATION_CLAIM = "ORGANIZATION_ALTERATION_CLAIM"
    DATE_ALTERATION_CLAIM = "DATE_ALTERATION_CLAIM"
    TENURE_OVERSTATEMENT_CLAIM = "TENURE_OVERSTATEMENT_CLAIM"
    UNDISCLOSED_METRIC_CLAIM = "UNDISCLOSED_METRIC_CLAIM"
    FALSE_INDEPENDENT_VERIFICATION_CLAIM = "FALSE_INDEPENDENT_VERIFICATION_CLAIM"
    PII_LEAKAGE = "PII_LEAKAGE"
    SCORE_IMMUTABILITY_BREACH = "SCORE_IMMUTABILITY_BREACH"
    AWKWARD_GENERATED_PHRASING = "AWKWARD_GENERATED_PHRASING"
    UNSUPPORTED_CLAIM_EXPANSION = "UNSUPPORTED_CLAIM_EXPANSION"
    DUPLICATE_BULLET_POINT = "DUPLICATE_BULLET_POINT"
    UNGROUNDED_RESPONSIBILITY_EXPANSION = "UNGROUNDED_RESPONSIBILITY_EXPANSION"
    INVALID_STAR_STRUCTURE_CLAIM = "INVALID_STAR_STRUCTURE_CLAIM"


class ViolationSeverity(str, Enum):
    """Severity classification for validation violations."""
    ERROR = "ERROR"      # Blocking violation requiring correction or human review
    WARNING = "WARNING"  # Non-blocking advisory finding


class InterviewQuestionCategory(str, Enum):
    """Categorization for structured interview coaching inquiries."""
    TECHNICAL = "TECHNICAL"
    ARCHITECTURE = "ARCHITECTURE"
    GAP_PROBE = "GAP_PROBE"
    METRIC_CHALLENGE = "METRIC_CHALLENGE"
    BEHAVIORAL = "BEHAVIORAL"


# =============================================================================
# Provenance Model
# =============================================================================

class EvidenceProvenanceRecord(BaseModel):
    """
    Lightweight record linking a claim or deliverable element to candidate evidence.
    Stores stable entity IDs and field paths rather than copied raw text snippets.
    """
    model_config = ConfigDict(frozen=True)

    provenance_id: str = Field(..., min_length=1, max_length=100, description="Unique UUID for this provenance entry.")
    target_entity_id: str = Field(..., min_length=1, max_length=100, description="Stable entity identifier (e.g. 'exp_0', 'proj_1', 'skill_Python').")
    source_field_path: str = Field(..., min_length=1, max_length=200, description="Field path reference (e.g. 'experience[0].responsibilities[1]').")
    provenance_tier: ProvenanceTier = Field(..., description="Evidence entry mechanism.")
    metric_status: MetricDisclosureStatus = Field(default=MetricDisclosureStatus.NOT_APPLICABLE)
    url_status: URLVerificationStatus = Field(default=URLVerificationStatus.NO_URL_PROVIDED)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Deliverable Schemas: Triage, Remediation, Resume Optimization & Interview Kit
# =============================================================================

class TriageAssessment(BaseModel):
    """
    Structured assessment emitted by the Triage & Diagnostics Agent.
    Provides diagnostic summaries and advisory workflow path suggestions.
    """
    model_config = ConfigDict(frozen=True)

    primary_bottleneck: str = Field(..., min_length=1, max_length=500, description="Core qualification or evidence bottleneck.")
    key_strengths_summary: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Top verified candidate qualifications.")
    critical_gaps_summary: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Priority missing or partial requirements.")
    recommended_focus_areas: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="High-yield development focus recommendations.")
    suggested_goal: UserWorkflowGoal = Field(..., description="Advisory workflow goal suggested by triage.")
    advisory_rationale: str = Field(..., min_length=1, max_length=1000, description="Explanation for suggested focus path.")


class UpskillingMilestone(BaseModel):
    """Specific phased upskilling milestone addressing target JD skill gaps."""
    model_config = ConfigDict(frozen=True)

    timeframe: str = Field(..., min_length=1, max_length=100, description="e.g. 'Days 1-30', 'Days 31-60', 'Days 61-90'.")
    target_missing_skills: tuple[str, ...] = Field(..., min_length=1, max_length=10, description="Target missing skills from employer requirements.")
    learning_objectives: tuple[str, ...] = Field(..., min_length=1, max_length=10, description="Specific conceptual and hands-on learning goals.")
    concrete_deliverable: str = Field(..., min_length=1, max_length=500, description="Verifiable tangible deliverable demonstrating mastery.")


class PortfolioProjectSpecification(BaseModel):
    """Engineering blueprint for a portfolio project addressing candidate evidence gaps."""
    model_config = ConfigDict(frozen=True)

    project_concept: str = Field(..., min_length=1, max_length=200, description="Title and conceptual description of the project.")
    architecture_overview: str = Field(..., min_length=1, max_length=1000, description="System design, components, and data flow.")
    targeted_jd_gaps: tuple[str, ...] = Field(..., min_length=1, max_length=10, description="Employer requirements demonstrated by this project.")
    recommended_tech_stack: tuple[str, ...] = Field(..., min_length=1, max_length=15, description="Recommended production-grade technologies and tools.")
    production_grade_checklist: tuple[str, ...] = Field(..., min_length=1, max_length=10, description="Testing, containerization, metrics, and CI/CD criteria.")


class CareerRemediationPlan(BaseModel):
    """Complete strategic upskilling plan and portfolio project recommendations."""
    model_config = ConfigDict(frozen=True)

    executive_summary: str = Field(..., min_length=1, max_length=1000, description="Strategic overview of candidate development trajectory.")
    milestones: tuple[UpskillingMilestone, ...] = Field(..., min_length=1, max_length=6, description="Phased learning milestones.")
    portfolio_projects: tuple[PortfolioProjectSpecification, ...] = Field(..., min_length=1, max_length=4, description="Engineering project blueprints.")
    estimated_readiness_timeline: str = Field(..., min_length=1, max_length=200, description="Estimated timeframe to achieve role alignment.")


class OptimizedBulletPoint(BaseModel):
    """
    Optimized resume bullet point using STAR framework, strictly grounded
    in verified candidate experience or project records.
    """
    model_config = ConfigDict(frozen=True)

    bullet_id: str = Field(..., min_length=1, max_length=100, description="Identifier for this bullet entry.")
    source_entity_id: str = Field(..., min_length=1, max_length=100, description="Reference to source experience or project record (e.g. 'exp_0').")
    source_field_path: str = Field(..., min_length=1, max_length=200, description="Field path to original evidence (e.g. 'experience[0].responsibilities[1]').")
    improved_star_bullet: str = Field(..., min_length=1, max_length=500, description="Reframed, high-impact STAR bullet point.")
    highlighted_skills: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Verified candidate skills prominently featured.")
    alignment_rationale: str = Field(..., min_length=1, max_length=400, description="Rationale for phrasing adjustments and keyword emphasis.")
    metric_disclosure: MetricDisclosureStatus = Field(
        default=MetricDisclosureStatus.NOT_APPLICABLE,
        description="Disclosure status if quantitative metrics are stated.",
    )
    provenance_label: str = Field(
        default="Rewritten from candidate-provided evidence",
        max_length=150,
        description="Evidence grounding classification: Source-linked to resume evidence, Rewritten from candidate-provided evidence, or Candidate-confirmed.",
    )
    bullet_structure: str = Field(
        default="Action & Scope",
        max_length=100,
        description="Structural classification: 'Full STAR (Action & Measurable Result)' or 'Action & Scope (Self-reported responsibility)'",
    )


class ResumeOptimizationReport(BaseModel):
    """Consolidated resume optimization analysis and tailored bullet points."""
    model_config = ConfigDict(frozen=True)

    target_role_alignment_summary: str = Field(..., min_length=1, max_length=1000, description="Strategic synthesis of resume alignment with role.")
    optimized_bullets: tuple[OptimizedBulletPoint, ...] = Field(..., min_length=1, max_length=20, description="Collection of reframed STAR bullet points.")
    section_reordering_recommendations: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Structural reordering advice.")
    ats_keyword_density_feedback: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Analysis of keyword presence and natural integration.")


class InterviewQuestion(BaseModel):
    """Tailored interview question targeting verified skills, evidence gaps, or self-reported metrics."""
    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., min_length=1, max_length=100, description="Unique question identifier.")
    category: InterviewQuestionCategory = Field(..., description="Question focus category.")
    question_text: str = Field(..., min_length=1, max_length=500, description="The interview inquiry.")
    interviewer_intent: str = Field(..., min_length=1, max_length=500, description="Core competencies or depth being assessed.")
    targeted_skill_or_gap: str = Field(..., min_length=1, max_length=100, description="Skill, metric, or requirement being evaluated.")
    suggested_talking_points: tuple[str, ...] = Field(..., min_length=1, max_length=10, description="Grounded talking points linking to candidate experience.")
    grounded_source_entity_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Entity ID of candidate experience/project cited in talking points if applicable.",
    )


class InterviewPreparationKit(BaseModel):
    """Role-specific technical and behavioral interview preparation package."""
    model_config = ConfigDict(frozen=True)

    role_context_overview: str = Field(..., min_length=1, max_length=1000, description="Synthesis of expected interview themes and standards.")
    technical_deep_dive_questions: tuple[InterviewQuestion, ...] = Field(..., min_length=1, max_length=10, description="Inquiries into verified core competencies.")
    gap_mitigation_questions: tuple[InterviewQuestion, ...] = Field(default_factory=tuple, max_length=10, description="Questions probing unevidenced or adjacent skills.")
    metric_challenge_questions: tuple[InterviewQuestion, ...] = Field(default_factory=tuple, max_length=10, description="Probes regarding self-reported metric claims.")
    recommended_questions_for_interviewer: tuple[str, ...] = Field(default_factory=tuple, max_length=10, description="Thoughtful questions for the candidate to ask.")


# =============================================================================
# Validation and Telemetry Models
# =============================================================================

class DeterministicValidationViolation(BaseModel):
    """Detailed record of a factual, immutability, or provenance validation violation."""
    model_config = ConfigDict(frozen=True)

    violation_type: ValidationViolationType = Field(..., description="Categorization of the violation.")
    severity: ViolationSeverity = Field(default=ViolationSeverity.ERROR, description="Violation severity level.")
    message: str = Field(..., min_length=1, max_length=500, description="Explanatory message describing the failure.")
    target_field: Optional[str] = Field(default=None, max_length=200, description="Field path where the violation was detected.")
    invalid_value: Optional[str] = Field(default=None, max_length=200, description="Sanitized, non-PII representation of the invalid token.")


class DeterministicValidationReport(BaseModel):
    """Consolidated report emitted by deterministic validators for an agent deliverable."""
    model_config = ConfigDict(frozen=True)

    report_id: str = Field(..., min_length=1, max_length=100, description="Unique report identifier.")
    is_valid: bool = Field(..., description="True if deliverable passed all deterministic checks with zero errors.")
    target_deliverable_type: str = Field(..., min_length=1, max_length=100, description="Schema type of the validated deliverable.")
    violations: tuple[DeterministicValidationViolation, ...] = Field(default_factory=tuple, description="List of detected violations.")
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowTraceEvent(BaseModel):
    """Telemetry trace event recording state transitions, latencies, and node activity."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1, max_length=100, description="Unique trace identifier.")
    node_name: str = Field(..., min_length=1, max_length=100, description="Workflow node that produced this event.")
    event_type: str = Field(..., min_length=1, max_length=100, description="e.g. 'NODE_START', 'NODE_END', 'VALIDATION_FAILURE'.")
    message: str = Field(..., min_length=1, max_length=500, description="Sanitized description of the trace event.")
    elapsed_ms: float = Field(default=0.0, ge=0.0, description="Elapsed execution duration in milliseconds.")
    token_count: Optional[int] = Field(default=None, ge=0, description="Token consumption if applicable.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuthoritativeMatchSnapshot(BaseModel):
    """
    Immutable canonical representation of the Phase 4 deterministic JobMatchResult.
    Locks all 16 authoritative scoring, dimensional, and evidence fields into
    immutable tuples and includes a deterministic SHA-256 integrity hash.
    Nodes in workflow state cannot mutate this representation.
    """
    model_config = ConfigDict(frozen=True)

    # 1. Deterministic Scoring & Classification
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Deterministic overall compatibility score.")
    fit_classification: CandidateFitLevel = Field(..., description="Overall fit classification.")
    evaluation_status: EvaluationStatus = Field(..., description="Assessment status.")
    active_evaluation_coverage: float = Field(..., ge=0.0, le=100.0, description="Criteria coverage percentage.")
    unassessed_reserve: float = Field(default=0.0, ge=0.0, le=100.0, description="Criteria reserve percentage.")

    # 2. Immutable Collections (Tuples replace mutable lists)
    match_breakdown: tuple[MatchBreakdown, ...] = Field(default_factory=tuple, description="Immutable dimensional breakdowns.")
    matched_skills: tuple[SkillMatch, ...] = Field(default_factory=tuple, description="Immutable verified technical skills.")
    partial_matches: tuple[SkillMatch, ...] = Field(default_factory=tuple, description="Immutable related technology matches.")
    missing_evidence: tuple[SkillGap, ...] = Field(default_factory=tuple, description="Immutable unevidenced requirement gaps.")
    unknown_evidence: tuple[SkillMatch, ...] = Field(default_factory=tuple, description="Immutable qualitative criteria.")
    project_evidence: tuple[ProjectEvidence, ...] = Field(default_factory=tuple, description="Immutable project audit records.")
    strengths: tuple[str, ...] = Field(default_factory=tuple, description="Immutable highlighted candidate strengths.")
    skill_gaps: tuple[SkillGap, ...] = Field(default_factory=tuple, description="Immutable skill gap learning paths.")
    recommendations: tuple[str, ...] = Field(default_factory=tuple, description="Immutable recommendations.")
    processing_warnings: tuple[str, ...] = Field(default_factory=tuple, description="Immutable processing diagnostics.")

    # 3. Canonical Job Metadata & Cryptographic Integrity Anchor
    job_metadata_canonical_json: str = Field(default="{}", description="Canonical JSON representation of job metadata.")
    canonical_json: str = Field(..., min_length=1, description="Deterministic canonical JSON of all 16 authoritative fields.")
    sha256_digest: str = Field(..., min_length=64, max_length=64, description="SHA-256 digest of canonical_json.")

    @classmethod
    def from_job_match_result(cls, result: Any) -> "AuthoritativeMatchSnapshot":
        """Construct an immutable AuthoritativeMatchSnapshot from a Phase 4 JobMatchResult."""
        import json
        from core.workflow.validators import (
            compute_match_result_digest,
            ensure_canonical_job_match_result,
            serialize_authoritative_match_result,
        )

        canonical_result = ensure_canonical_job_match_result(result)
        canonical_json = serialize_authoritative_match_result(canonical_result)
        digest = compute_match_result_digest(canonical_json)
        meta_json = json.dumps(canonical_result.job_metadata or {}, sort_keys=True, separators=(',', ':'))

        def _to_dicts(seq):
            return tuple(item.model_dump() if hasattr(item, "model_dump") else item for item in seq)

        return cls(
            overall_score=canonical_result.overall_score,
            fit_classification=canonical_result.fit_classification,
            evaluation_status=canonical_result.evaluation_status,
            active_evaluation_coverage=canonical_result.active_evaluation_coverage,
            unassessed_reserve=canonical_result.unassessed_reserve,
            match_breakdown=_to_dicts(canonical_result.match_breakdown),
            matched_skills=_to_dicts(canonical_result.matched_skills),
            partial_matches=_to_dicts(canonical_result.partial_matches),
            missing_evidence=_to_dicts(canonical_result.missing_evidence),
            unknown_evidence=_to_dicts(canonical_result.unknown_evidence),
            project_evidence=_to_dicts(canonical_result.project_evidence),
            strengths=tuple(canonical_result.strengths),
            skill_gaps=_to_dicts(canonical_result.skill_gaps),
            recommendations=tuple(canonical_result.recommendations),
            processing_warnings=tuple(canonical_result.processing_warnings),
            job_metadata_canonical_json=meta_json,
            canonical_json=canonical_json,
            sha256_digest=digest,
        )

    def restore_job_match_result(self) -> JobMatchResult:
        """
        Deterministically reconstruct an uncorrupted, fresh JobMatchResult from the canonical JSON.
        Guarantees that downstream comparisons operate on an unmutated baseline.
        """
        from models.matching import JobMatchResult
        return JobMatchResult.model_validate_json(self.canonical_json)

    @property
    def job_metadata(self) -> Any:
        """
        Immutable read-only mapping of job metadata preventing in-place dictionary modification.
        Any item assignment or mutation attempt raises TypeError.
        """
        import json
        import types
        return types.MappingProxyType(json.loads(self.job_metadata_canonical_json))

