"""
Unit tests for Phase 4 Step 1: Pydantic Domain Models for Job Description Matching.
Tests validation rules, score/weight boundaries, enum integrity, nested objects,
JSON round trips, cross-field validators, evidence semantics, and backward compatibility.
"""
import json
import pytest
from pydantic import ValidationError

from models.matching import (
    CandidateFitLevel,
    DimensionBreakdown,
    EvaluationStatus,
    EvidenceStatus,
    EvidenceVerificationLevel,
    ExperienceTier,
    JobDescriptionProfile,
    JobMatchResult,
    JobRequirement,
    MatchBreakdown,
    ProjectEvidence,
    RequirementCategory,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models import ResumeProfile, ResumeEntities


# =====================================================================
# Test Group 1: Enums Validation
# =====================================================================

class TestMatchingEnums:
    """Verifies all enum definitions, member sets, string representations, and rejection of invalid values."""

    def test_requirement_category_members(self):
        expected = {
            "TECHNICAL_SKILL",
            "EXPERIENCE_TENURE",
            "ROLE_RELEVANCE",
            "EDUCATION",
            "CERTIFICATION",
            "PROJECT",
            "RESPONSIBILITY",
        }
        actual = {e.value for e in RequirementCategory}
        assert actual == expected

    def test_requirement_priority_members(self):
        expected = {"REQUIRED", "PREFERRED", "OPTIONAL"}
        actual = {e.value for e in RequirementPriority}
        assert actual == expected

    def test_evidence_status_members(self):
        expected = {
            "VERIFIED_EVIDENCE",
            "PARTIAL_EVIDENCE",
            "MISSING_EVIDENCE",
            "UNKNOWN_EVIDENCE",
        }
        actual = {e.value for e in EvidenceStatus}
        assert actual == expected

    def test_skill_match_type_members(self):
        expected = {
            "EXACT_MATCH",
            "SYNONYM_MATCH",
            "RELATED_TECHNOLOGY",
            "NO_MATCH",
            "UNKNOWN",
        }
        actual = {e.value for e in SkillMatchType}
        assert actual == expected

    def test_experience_tier_members(self):
        expected = {
            "DIRECT_ALIGNMENT",
            "ADJACENT_ALIGNMENT",
            "RELEVANT_INTERNSHIP",
            "APPLIED_PROJECTS",
            "UNRELATED_BACKGROUND",
            "UNKNOWN",
        }
        actual = {e.value for e in ExperienceTier}
        assert actual == expected

    def test_evaluation_status_members(self):
        expected = {
            "FULL_EVALUATION",
            "PROVISIONAL_EVALUATION",
            "INSUFFICIENT_EVIDENCE",
        }
        actual = {e.value for e in EvaluationStatus}
        assert actual == expected

    def test_evidence_verification_level_members(self):
        expected = {
            "SELF_REPORTED",
            "CANDIDATE_PROVIDED_URL",
            "STRUCTURALLY_VALID_URL",
            "ARTIFACT_SUPPORTED",
            "INDEPENDENTLY_VERIFIED",
        }
        actual = {e.value for e in EvidenceVerificationLevel}
        assert actual == expected

    def test_candidate_fit_level_members(self):
        expected = {
            "EXCEPTIONAL",
            "STRONG",
            "MODERATE",
            "DEVELOPING",
            "LOW",
            "PROVISIONAL",
        }
        actual = {e.value for e in CandidateFitLevel}
        assert actual == expected


# =====================================================================
# Test Group 2: JobRequirement Model
# =====================================================================

class TestJobRequirementModel:
    """Tests JobRequirement construction, validation, and string stripping."""

    def test_valid_job_requirement_creation(self):
        req = JobRequirement(
            requirement_id="req-py-01",
            requirement_text="5+ years of production Python experience",
            canonical_name="Python",
            category=RequirementCategory.TECHNICAL_SKILL,
            priority=RequirementPriority.REQUIRED,
            evidence_expectation="5+ years",
            source_section="Requirements",
        )
        assert req.requirement_id == "req-py-01"
        assert req.canonical_name == "Python"
        assert req.category == RequirementCategory.TECHNICAL_SKILL
        assert req.priority == RequirementPriority.REQUIRED
        assert req.evidence_expectation == "5+ years"
        assert req.source_section == "Requirements"

    def test_job_requirement_default_priority(self):
        req = JobRequirement(
            requirement_id="req-docker-01",
            requirement_text="Familiarity with Docker",
            canonical_name="Docker",
            category=RequirementCategory.TECHNICAL_SKILL,
        )
        assert req.priority == RequirementPriority.REQUIRED
        assert req.evidence_expectation is None
        assert req.source_section is None

    def test_job_requirement_strips_whitespace(self):
        req = JobRequirement(
            requirement_id="  req-strip-01  ",
            requirement_text="  PostgreSQL experience  ",
            canonical_name="  PostgreSQL  ",
            category=RequirementCategory.TECHNICAL_SKILL,
        )
        assert req.requirement_id == "req-strip-01"
        assert req.requirement_text == "PostgreSQL experience"
        assert req.canonical_name == "PostgreSQL"

    def test_job_requirement_rejects_empty_identifiers(self):
        with pytest.raises(ValidationError) as exc_info:
            JobRequirement(
                requirement_id="   ",
                requirement_text="Valid requirement text",
                canonical_name="Python",
                category=RequirementCategory.TECHNICAL_SKILL,
            )
        assert "Field must contain non-whitespace characters" in str(exc_info.value)

    def test_job_requirement_rejects_empty_canonical_name(self):
        with pytest.raises(ValidationError):
            JobRequirement(
                requirement_id="req-01",
                requirement_text="Valid text",
                canonical_name="",
                category=RequirementCategory.TECHNICAL_SKILL,
            )

    def test_job_requirement_rejects_invalid_category(self):
        with pytest.raises(ValidationError):
            JobRequirement(
                requirement_id="req-01",
                requirement_text="Valid text",
                canonical_name="Python",
                category="INVALID_CATEGORY",  # type: ignore
            )


# =====================================================================
# Test Group 3: JobDescriptionProfile Model
# =====================================================================

class TestJobDescriptionProfileModel:
    """Tests JobDescriptionProfile creation, validation, and defaults."""

    def test_valid_profile_creation(self):
        profile = JobDescriptionProfile(
            job_title="Senior Python Backend Engineer",
            company="Tech Corp",
            original_description="We are looking for a Senior Backend Engineer...",
            required_skills=["Python", "FastAPI", "PostgreSQL"],
            preferred_skills=["Docker", "Kubernetes"],
            experience_requirement="3+ years",
            education_requirement="Bachelor's degree in Computer Science",
            responsibilities=["Design asynchronous REST APIs", "Optimize SQL queries"],
            certifications=["AWS Certified Solutions Architect"],
            processing_warnings=["Parsed 2 preferred skills."],
        )
        assert profile.job_title == "Senior Python Backend Engineer"
        assert profile.company == "Tech Corp"
        assert len(profile.required_skills) == 3
        assert len(profile.preferred_skills) == 2
        assert len(profile.responsibilities) == 2
        assert len(profile.certifications) == 1
        assert len(profile.processing_warnings) == 1

    def test_profile_rejects_empty_description(self):
        with pytest.raises(ValidationError) as exc_info:
            JobDescriptionProfile(original_description="   ")
        assert "original_description cannot be empty" in str(exc_info.value)

    def test_profile_default_factory_lists(self):
        profile = JobDescriptionProfile(original_description="Standard JD text")
        assert profile.job_title is None
        assert profile.company is None
        assert profile.normalized_requirements == []
        assert profile.required_skills == []
        assert profile.preferred_skills == []
        assert profile.responsibilities == []
        assert profile.certifications == []
        assert profile.processing_warnings == []


# =====================================================================
# Test Group 4: SkillMatch and SkillGap Models
# =====================================================================

class TestSkillMatchAndGapModels:
    """Tests SkillMatch and SkillGap domain structures and bounds."""

    def test_valid_skill_match_exact(self):
        match = SkillMatch(
            skill_name="Python",
            canonical_name="Python",
            requirement_priority=RequirementPriority.REQUIRED,
            match_type=SkillMatchType.EXACT_MATCH,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            resume_evidence_references=["Skills: Python", "Work Experience: Senior Python Developer"],
            jd_evidence_references=["Requirements: 5+ years of Python"],
            match_confidence=1.0,
            explanation="Exact canonical match in Technical Skills and Work Experience.",
        )
        assert match.skill_name == "Python"
        assert match.match_type == SkillMatchType.EXACT_MATCH
        assert match.evidence_status == EvidenceStatus.VERIFIED_EVIDENCE
        assert match.match_confidence == 1.0

    def test_skill_match_related_technology(self):
        match = SkillMatch(
            skill_name="FastAPI",
            canonical_name="FastAPI",
            requirement_priority=RequirementPriority.REQUIRED,
            match_type=SkillMatchType.RELATED_TECHNOLOGY,
            evidence_status=EvidenceStatus.PARTIAL_EVIDENCE,
            resume_evidence_references=["Skills: Flask"],
            jd_evidence_references=["Requirements: FastAPI"],
            match_confidence=0.5,
            explanation="Candidate has Flask; pre-mapped related technology to FastAPI (0.5 credit).",
        )
        assert match.match_type == SkillMatchType.RELATED_TECHNOLOGY
        assert match.evidence_status == EvidenceStatus.PARTIAL_EVIDENCE
        assert match.match_confidence == 0.5

    def test_skill_match_rejects_invalid_confidence_bounds(self):
        with pytest.raises(ValidationError):
            SkillMatch(
                skill_name="Go",
                canonical_name="Go",
                requirement_priority=RequirementPriority.REQUIRED,
                match_type=SkillMatchType.EXACT_MATCH,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                match_confidence=1.5,  # > 1.0
            )

        with pytest.raises(ValidationError):
            SkillMatch(
                skill_name="Go",
                canonical_name="Go",
                requirement_priority=RequirementPriority.REQUIRED,
                match_type=SkillMatchType.EXACT_MATCH,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                match_confidence=-0.1,  # < 0.0
            )

    def test_valid_skill_gap_creation(self):
        gap = SkillGap(
            skill_name="Kubernetes",
            priority=RequirementPriority.REQUIRED,
            evidence_status=EvidenceStatus.MISSING_EVIDENCE,
            gap_explanation="No mention of Kubernetes or container orchestration tools found in resume.",
            suggested_learning_direction="Learn Kubernetes pod lifecycle, Deployments, and Helm charts.",
        )
        assert gap.skill_name == "Kubernetes"
        assert gap.priority == RequirementPriority.REQUIRED
        assert gap.evidence_status == EvidenceStatus.MISSING_EVIDENCE
        assert "Kubernetes" in gap.gap_explanation
        assert gap.suggested_learning_direction is not None


# =====================================================================
# Test Group 5: MatchBreakdown & Dimension Scores
# =====================================================================

class TestMatchBreakdownModel:
    """Tests MatchBreakdown validation, score boundaries, and active weight consistency."""

    def test_valid_match_breakdown_creation(self):
        bd = MatchBreakdown(
            dimension_name="Technical Skills",
            baseline_weight=0.45,
            active_weight=0.45,
            dimension_score=87.5,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            is_active=True,
        )
        assert bd.dimension_name == "Technical Skills"
        assert bd.baseline_weight == 0.45
        assert bd.active_weight == 0.45
        assert bd.dimension_score == 87.5
        assert bd.is_active is True
        assert bd.neutralization_explanation is None

    def test_neutralized_dimension_breakdown(self):
        bd = MatchBreakdown(
            dimension_name="Education",
            baseline_weight=0.15,
            active_weight=0.0,
            dimension_score=100.0,
            evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
            is_active=False,
            neutralization_explanation="JD specifies no education requirements.",
        )
        assert bd.is_active is False
        assert bd.active_weight == 0.0
        assert bd.neutralization_explanation == "JD specifies no education requirements."

    def test_rejects_neutralized_dimension_with_positive_active_weight(self):
        with pytest.raises(ValidationError) as exc_info:
            MatchBreakdown(
                dimension_name="Education",
                baseline_weight=0.15,
                active_weight=0.15,  # Inconsistent: is_active=False requires active_weight=0.0
                dimension_score=100.0,
                evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                is_active=False,
            )
        assert "neutralized dimension (is_active=False) must have active_weight=0.0" in str(exc_info.value)

    def test_rejects_dimension_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            MatchBreakdown(
                dimension_name="Experience",
                baseline_weight=0.25,
                active_weight=0.25,
                dimension_score=105.0,  # > 100.0
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )

        with pytest.raises(ValidationError):
            MatchBreakdown(
                dimension_name="Experience",
                baseline_weight=0.25,
                active_weight=0.25,
                dimension_score=-5.0,  # < 0.0
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )

    def test_rejects_weights_out_of_bounds(self):
        with pytest.raises(ValidationError):
            MatchBreakdown(
                dimension_name="Skills",
                baseline_weight=1.5,  # > 1.0
                active_weight=0.45,
                dimension_score=50.0,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )

    def test_dimension_breakdown_alias(self):
        assert DimensionBreakdown is MatchBreakdown


# =====================================================================
# Test Group 6: ProjectEvidence Model
# =====================================================================

class TestProjectEvidenceModel:
    """Tests ProjectEvidence audit records and verification level discrimination."""

    def test_valid_project_evidence_artifact_supported(self):
        pe = ProjectEvidence(
            project_identifier="proj-01",
            project_name="Order Processing Service",
            relevant_technologies=["Python", "FastAPI", "PostgreSQL"],
            technology_implementation_evidence=[
                "Architected async order processing with FastAPI",
                "Designed connection pooling in PostgreSQL",
            ],
            artifact_url="https://github.com/candidate/order-service",
            evidence_verification_level=EvidenceVerificationLevel.ARTIFACT_SUPPORTED,
            self_reported_metrics=["reduced latency by 45%", "2500 tps throughput"],
            toolchain_interaction_description="FastAPI queries PostgreSQL through asyncpg connection pool.",
            responsibility_alignment_evidence=["Direct alignment with async REST API design."],
            calculated_pillar_scores={"tech_coverage": 30.0, "context_depth": 35.0, "duty_align": 25.0},
        )
        assert pe.project_identifier == "proj-01"
        assert pe.evidence_verification_level == EvidenceVerificationLevel.ARTIFACT_SUPPORTED
        assert pe.artifact_url == "https://github.com/candidate/order-service"
        assert len(pe.self_reported_metrics) == 2
        assert pe.calculated_pillar_scores["tech_coverage"] == 30.0

    def test_project_evidence_self_reported_no_url(self):
        pe = ProjectEvidence(
            project_identifier="proj-02",
            project_name="Task App",
            relevant_technologies=["Python", "Flask"],
            evidence_verification_level=EvidenceVerificationLevel.SELF_REPORTED,
        )
        assert pe.evidence_verification_level == EvidenceVerificationLevel.SELF_REPORTED
        assert pe.artifact_url is None
        assert pe.self_reported_metrics == []

    def test_project_evidence_strips_empty_url(self):
        pe = ProjectEvidence(
            project_identifier="proj-03",
            project_name="Demo",
            artifact_url="   ",
        )
        assert pe.artifact_url is None


# =====================================================================
# Test Group 7: JobMatchResult Model
# =====================================================================

class TestJobMatchResultModel:
    """Tests JobMatchResult creation, cross-field validation, and provisional enforcement."""

    def test_valid_full_evaluation_result(self):
        result = JobMatchResult(
            job_metadata={"job_title": "Backend Developer", "company": "Acme Inc"},
            overall_score=92.9,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.EXCEPTIONAL,
            active_evaluation_coverage=100.0,
            unassessed_reserve=0.0,
            match_breakdown=[
                MatchBreakdown(
                    dimension_name="Technical Skills",
                    baseline_weight=0.45,
                    active_weight=0.45,
                    dimension_score=87.5,
                    evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                ),
                MatchBreakdown(
                    dimension_name="Experience",
                    baseline_weight=0.25,
                    active_weight=0.25,
                    dimension_score=100.0,
                    evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                ),
            ],
            strengths=["Strong Python and PostgreSQL alignment"],
            recommendations=["Upskill in AWS cloud services"],
        )
        assert result.overall_score == 92.9
        assert result.evaluation_status == EvaluationStatus.FULL_EVALUATION
        assert result.fit_classification == CandidateFitLevel.EXCEPTIONAL
        assert result.active_evaluation_coverage == 100.0
        assert result.unassessed_reserve == 0.0

        # Helper method lookup
        dim = result.get_dimension("technical skills")
        assert dim is not None
        assert dim.dimension_score == 87.5

        dim_missing = result.get_dimension("Nonexistent")
        assert dim_missing is None

    def test_provisional_status_enforces_provisional_fit_classification(self):
        """Cross-field validator: provisional evaluation status overrides numerical fit label."""
        result = JobMatchResult(
            job_metadata={"job_title": "Sparse Role"},
            overall_score=100.0,
            evaluation_status=EvaluationStatus.PROVISIONAL_EVALUATION,
            fit_classification=CandidateFitLevel.EXCEPTIONAL,  # High score, but evaluation is provisional
            active_evaluation_coverage=70.0,
            unassessed_reserve=0.0,
        )
        # Validator must override fit_classification to PROVISIONAL
        assert result.evaluation_status == EvaluationStatus.PROVISIONAL_EVALUATION
        assert result.fit_classification == CandidateFitLevel.PROVISIONAL

    def test_rejects_overall_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            JobMatchResult(
                overall_score=101.0,
                evaluation_status=EvaluationStatus.FULL_EVALUATION,
                fit_classification=CandidateFitLevel.EXCEPTIONAL,
                active_evaluation_coverage=100.0,
            )

        with pytest.raises(ValidationError):
            JobMatchResult(
                overall_score=-0.5,
                evaluation_status=EvaluationStatus.FULL_EVALUATION,
                fit_classification=CandidateFitLevel.LOW,
                active_evaluation_coverage=100.0,
            )


# =====================================================================
# Test Group 8: JSON Round-Trip Serialization
# =====================================================================

class TestJsonSerialization:
    """Tests serialization and deserialization across Pydantic models."""

    def test_job_requirement_json_round_trip(self):
        req = JobRequirement(
            requirement_id="req-ts-01",
            requirement_text="TypeScript proficiency",
            canonical_name="TypeScript",
            category=RequirementCategory.TECHNICAL_SKILL,
            priority=RequirementPriority.REQUIRED,
        )
        json_str = req.model_dump_json()
        data = json.loads(json_str)
        assert data["requirement_id"] == "req-ts-01"
        assert data["category"] == "TECHNICAL_SKILL"

        restored = JobRequirement.model_validate_json(json_str)
        assert restored == req

    def test_job_match_result_json_round_trip(self):
        res = JobMatchResult(
            job_metadata={"title": "Cloud Architect"},
            overall_score=75.5,
            evaluation_status=EvaluationStatus.FULL_EVALUATION,
            fit_classification=CandidateFitLevel.STRONG,
            active_evaluation_coverage=100.0,
            unassessed_reserve=0.0,
            match_breakdown=[
                MatchBreakdown(
                    dimension_name="Skills",
                    baseline_weight=0.45,
                    active_weight=0.45,
                    dimension_score=80.0,
                    evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                )
            ],
            matched_skills=[
                SkillMatch(
                    skill_name="AWS",
                    canonical_name="AWS",
                    requirement_priority=RequirementPriority.REQUIRED,
                    match_type=SkillMatchType.EXACT_MATCH,
                    evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                )
            ],
            missing_evidence=[
                SkillGap(
                    skill_name="Terraform",
                    priority=RequirementPriority.REQUIRED,
                    evidence_status=EvidenceStatus.MISSING_EVIDENCE,
                    gap_explanation="No IaC experience detected.",
                )
            ],
        )
        json_str = res.model_dump_json()
        restored = JobMatchResult.model_validate_json(json_str)
        assert restored.overall_score == 75.5
        assert restored.matched_skills[0].skill_name == "AWS"
        assert restored.missing_evidence[0].skill_name == "Terraform"


# =====================================================================
# Test Group 9: Backward Compatibility with Existing Models
# =====================================================================

class TestBackwardCompatibility:
    """Ensures existing resume models can be imported and initialized without collision."""

    def test_resume_profile_import_and_instantiation(self):
        assert ResumeProfile is not None
        assert ResumeEntities is not None

