"""
Unit tests for Phase 5.2 Deterministic Tools package:
EvidenceResolverTool, SkillVerifierTool, TenureVerifierTool,
MetricVerifierTool, PIIScannerTool, and ScoreIntegrityTool.
"""
import pytest

from core.matching.exceptions import ScoreImmutabilityError
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    EvidenceStatus,
    JobDescriptionProfile,
    JobMatchResult,
    MatchBreakdown,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models.resume_entities import (
    ExtractedSkill,
    ProjectRecord,
    SkillCategory,
    WorkExperience,
)
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from models.workflow import (
    AuthoritativeMatchSnapshot,
    ClaimContext,
    MetricDisclosureStatus,
    ProvenanceTier,
    ValidationViolationType,
)
from tools import (
    EvidenceResolverTool,
    MetricVerifierTool,
    PIIScannerTool,
    ScoreIntegrityTool,
    SkillVerifierTool,
    TenureVerifierTool,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_profile() -> ResumeProfile:
    """Minimal candidate profile with verified skills, experience, and projects."""
    return ResumeProfile(
        filename="candidate_resume.pdf",
        page_count=1,
        raw_character_count=800,
        cleaned_character_count=780,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(name="Python", original_mention="Python 3.11", category=SkillCategory.PROGRAMMING_LANGUAGES),
            ExtractedSkill(name="PostgreSQL", original_mention="PostgreSQL database", category=SkillCategory.DATABASES),
            ExtractedSkill(name="Docker", original_mention="Docker", category=SkillCategory.DEVOPS_AND_INFRASTRUCTURE),
        ],
        experience=[
            WorkExperience(
                job_title="Software Engineer",
                company="Nexus Tech",
                employment_start="2022",
                employment_end="2024",
                responsibilities=[
                    "Built asynchronous microservices in Python.",
                    "Optimized SQL query performance across PostgreSQL clusters.",
                ],
                technologies=["Python", "PostgreSQL", "Docker"],
                original_text="Software Engineer @ Nexus Tech (2022-2024)",
            )
        ],
        projects=[
            ProjectRecord(
                project_name="Data Ingestion Engine",
                description="Engineered streaming ETL pipeline with Python.",
                technologies=["Python"],
                original_text="Data Ingestion Engine with Python",
            )
        ],
    )


@pytest.fixture
def sample_match_result() -> JobMatchResult:
    """Authoritative JobMatchResult with populated dimensions and skills."""
    return JobMatchResult(
        overall_score=80.0,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=100.0,
        unassessed_reserve=0.0,
        match_breakdown=[
            MatchBreakdown(
                dimension_name="Technical Skills",
                dimension_score=85.0,
                baseline_weight=1.0,
                active_weight=1.0,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )
        ],
        matched_skills=[
            SkillMatch(
                skill_name="Python",
                canonical_name="Python",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )
        ],
        skill_gaps=[
            SkillGap(
                skill_name="Kubernetes",
                priority=RequirementPriority.REQUIRED,
                gap_explanation="Kubernetes requirement unevidenced in candidate resume.",
                suggested_learning_direction="Container orchestration",
            )
        ],
    )


# =============================================================================
# 1. Evidence Resolver Tool Tests
# =============================================================================

class TestEvidenceResolverTool:
    """Verifies that EvidenceResolverTool correctly resolves entity references."""

    def test_resolve_valid_experience_record(self, sample_profile):
        tool = EvidenceResolverTool()
        is_res, err, val = tool.resolve(
            entity_id="exp_0",
            field_path="experience[0].responsibilities[0]",
            profile=sample_profile,
        )
        assert is_res is True
        assert err is None
        assert "asynchronous microservices" in val

    def test_resolve_valid_project_record(self, sample_profile):
        tool = EvidenceResolverTool()
        is_res, err, val = tool.resolve(
            entity_id="proj_0",
            field_path="projects[0].description",
            profile=sample_profile,
        )
        assert is_res is True
        assert err is None
        assert "streaming ETL pipeline" in val

    def test_resolve_out_of_bounds_index_rejected(self, sample_profile):
        tool = EvidenceResolverTool()
        is_res, err, _ = tool.resolve(
            entity_id="exp_99",
            field_path="experience[99].job_title",
            profile=sample_profile,
        )
        assert is_res is False
        assert "out of bounds" in err

    def test_validate_provenance_tier_catches_misclassified_clarifications(self):
        tool = EvidenceResolverTool()
        viol = tool.validate_provenance_tier(
            tier=ProvenanceTier.RESUME_EXTRACTED,
            source_path="user_clarifications.added_skill",
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.MISSING_EVIDENCE_REFERENCE


# =============================================================================
# 2. Skill Verifier Tool Tests
# =============================================================================

class TestSkillVerifierTool:
    """Verifies candidate skill-claim whitelist and advisory recommendation distinction."""

    def test_candidate_assertion_for_verified_skill_passes(self, sample_profile):
        tool = SkillVerifierTool()
        viol = tool.validate_candidate_assertion(
            skill_name="Python",
            profile=sample_profile,
        )
        assert viol is None

    def test_candidate_assertion_for_synonym_passes(self, sample_profile):
        tool = SkillVerifierTool()
        # Postgres maps to PostgreSQL in synonym taxonomy
        viol = tool.validate_candidate_assertion(
            skill_name="Postgres",
            profile=sample_profile,
        )
        assert viol is None

    def test_candidate_assertion_for_unverified_skill_rejected(self, sample_profile):
        tool = SkillVerifierTool()
        viol = tool.validate_candidate_assertion(
            skill_name="Kubernetes",
            profile=sample_profile,
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.UNGROUNDED_SKILL_CLAIM
        assert "Kubernetes" in viol.message

    def test_advisory_recommendation_citing_missing_skill_passes(self, sample_profile):
        tool = SkillVerifierTool()
        # Advisory context allows citing missing requirements in learning roadmaps
        viol = tool.validate_advisory_recommendation(
            skill_name="Kubernetes",
            profile=sample_profile,
        )
        assert viol is None

    def test_enforces_claim_context_separation(self, sample_profile):
        tool = SkillVerifierTool()
        # Candidate assertion of unverified skill fails
        v_assert = tool.validate("FastAPI", ClaimContext.CANDIDATE_ASSERTION, sample_profile)
        assert v_assert is not None

        # Advisory recommendation of same skill passes
        v_advisory = tool.validate("FastAPI", ClaimContext.ADVISORY_RECOMMENDATION, sample_profile)
        assert v_advisory is None


# =============================================================================
# 3. Tenure Verifier Tool Tests
# =============================================================================

class TestTenureVerifierTool:
    """Verifies employment metadata fidelity and tenure representation."""

    def test_verbatim_title_and_company_passes(self, sample_profile):
        tool = TenureVerifierTool()
        source_exp = sample_profile.experience[0]
        viols = tool.validate_experience_metadata(
            source_exp=source_exp,
            claimed_title="Software Engineer",
            claimed_company="Nexus Tech",
        )
        assert len(viols) == 0

    def test_altered_job_title_rejected(self, sample_profile):
        tool = TenureVerifierTool()
        source_exp = sample_profile.experience[0]
        viols = tool.validate_experience_metadata(
            source_exp=source_exp,
            claimed_title="VP of Engineering",
            claimed_company="Nexus Tech",
        )
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.TITLE_ALTERATION_CLAIM

    def test_altered_company_rejected(self, sample_profile):
        tool = TenureVerifierTool()
        source_exp = sample_profile.experience[0]
        viols = tool.validate_experience_metadata(
            source_exp=source_exp,
            claimed_title="Software Engineer",
            claimed_company="Acme BigCorp",
        )
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.ORGANIZATION_ALTERATION_CLAIM

    def test_project_represented_as_employment_tenure_rejected(self, sample_profile):
        tool = TenureVerifierTool()
        source_proj = sample_profile.projects[0]
        viol = tool.validate_no_project_as_employment(
            source_record=source_proj,
            claimed_as_employment=True,
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.TENURE_OVERSTATEMENT_CLAIM


# =============================================================================
# 4. Metric Verifier Tool Tests
# =============================================================================

class TestMetricVerifierTool:
    """Verifies quantitative claim disclosure rules and false audit rejection."""

    def test_quantitative_claim_with_self_reported_disclosure_passes(self):
        tool = MetricVerifierTool()
        viols = tool.validate_metrics(
            text="Reduced query latency by 40% through indexing.",
            disclosure_status=MetricDisclosureStatus.SELF_REPORTED_METRIC,
        )
        assert len(viols) == 0

    def test_quantitative_claim_without_disclosure_rejected(self):
        tool = MetricVerifierTool()
        viols = tool.validate_metrics(
            text="Boosted system throughput by 50% across microservices.",
            disclosure_status=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.UNDISCLOSED_METRIC_CLAIM

    def test_false_third_party_verification_statement_rejected(self):
        tool = MetricVerifierTool()
        viols = tool.validate_metrics(
            text="Independently verified code performance in production benchmark.",
            disclosure_status=MetricDisclosureStatus.SELF_REPORTED_METRIC,
        )
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.FALSE_INDEPENDENT_VERIFICATION_CLAIM


# =============================================================================
# 5. PII Scanner Tool Tests
# =============================================================================

class TestPIIScannerTool:
    """Verifies candidate PII detection and redaction."""

    def test_detects_candidate_email(self):
        tool = PIIScannerTool()
        viols = tool.scan("Candidate email: user@testcompany.org for contact.")
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.PII_LEAKAGE
        assert viols[0].invalid_value == "[REDACTED_EMAIL]"

    def test_detects_candidate_phone(self):
        tool = PIIScannerTool()
        viols = tool.scan("Phone: 555-019-2834 available for calls.")
        assert len(viols) == 1
        assert viols[0].violation_type == ValidationViolationType.PII_LEAKAGE
        assert viols[0].invalid_value == "[REDACTED_PHONE]"

    def test_clean_technical_text_has_zero_violations(self):
        tool = PIIScannerTool()
        viols = tool.scan("Architected asynchronous processing engine using Python and PostgreSQL.")
        assert len(viols) == 0


# =============================================================================
# 6. Score Integrity Tool Tests
# =============================================================================

class TestScoreIntegrityTool:
    """Verifies baseline snapshot creation, digest computation, and score immutability."""

    def test_create_snapshot_returns_authoritative_match_snapshot(self, sample_match_result):
        snap = ScoreIntegrityTool.create_snapshot(sample_match_result)
        assert isinstance(snap, AuthoritativeMatchSnapshot)
        assert len(snap.sha256_digest) == 64
        assert snap.overall_score == sample_match_result.overall_score

    def test_compute_digest_is_deterministic(self, sample_match_result):
        d1 = ScoreIntegrityTool.compute_digest(sample_match_result)
        d2 = ScoreIntegrityTool.compute_digest(sample_match_result)
        assert d1 == d2
        assert len(d1) == 64

    def test_verify_immutability_passes_for_intact_candidate(self, sample_match_result):
        snap = ScoreIntegrityTool.create_snapshot(sample_match_result)
        candidate = snap.restore_job_match_result()
        ScoreIntegrityTool.verify_immutability(
            baseline=snap,
            candidate=candidate,
            expected_sha256=snap.sha256_digest,
        )

    def test_verify_immutability_fails_for_tampered_candidate(self, sample_match_result):
        snap = ScoreIntegrityTool.create_snapshot(sample_match_result)
        tampered = sample_match_result.model_copy(update={"overall_score": 99.0}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            ScoreIntegrityTool.verify_immutability(
                baseline=snap,
                candidate=tampered,
                expected_sha256=snap.sha256_digest,
            )

    def test_intercept_update_blocks_protected_keys(self):
        with pytest.raises(ScoreImmutabilityError):
            ScoreIntegrityTool.intercept_update("node_1", {"match_result": "tampered"})

        with pytest.raises(ScoreImmutabilityError):
            ScoreIntegrityTool.intercept_update("node_1", {"baseline_sha256": "fake_hash"})

