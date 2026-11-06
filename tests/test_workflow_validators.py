"""
Fixture-based unit tests for Phase 5.1 deterministic validators:
Evidence reference resolution, candidate skill grounding, entity/tenure fidelity,
metric disclosures, and PII leakage prevention.
"""
import pytest

from core.workflow.validators import (
    CandidateSkillGroundingValidator,
    DeterministicWorkflowValidator,
    EntityTenureValidator,
    EvidenceReferenceResolver,
    MetricGroundingValidator,
    PIILeakageValidator,
)
from models.resume_entities import (
    ExtractedSkill,
    ProjectRecord,
    SkillCategory,
    WorkExperience,
)
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from models.workflow import (
    CareerRemediationPlan,
    ClaimContext,
    InterviewPreparationKit,
    InterviewQuestion,
    InterviewQuestionCategory,
    MetricDisclosureStatus,
    OptimizedBulletPoint,
    PortfolioProjectSpecification,
    ProvenanceTier,
    ResumeOptimizationReport,
    UpskillingMilestone,
    URLVerificationStatus,
    ValidationViolationType,
    ViolationSeverity,
)


# =============================================================================
# Test Fixture Setup
# =============================================================================

@pytest.fixture
def sample_candidate_profile() -> ResumeProfile:
    """Standardized candidate profile with known skills, experience, and projects."""
    return ResumeProfile(
        filename="alice_smith_resume.pdf",
        page_count=2,
        raw_character_count=1200,
        cleaned_character_count=1150,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(name="Python", original_mention="Python 3.11", category=SkillCategory.PROGRAMMING_LANGUAGES),
            ExtractedSkill(name="PostgreSQL", original_mention="Postgres database", category=SkillCategory.DATABASES),
            ExtractedSkill(name="Flask", original_mention="Flask microframework", category=SkillCategory.WEB_DEVELOPMENT),
            ExtractedSkill(name="Docker", original_mention="Docker containers", category=SkillCategory.DEVOPS_AND_INFRASTRUCTURE),
        ],
        experience=[
            WorkExperience(
                job_title="Backend Software Engineer",
                company="Acme Cloud Systems",
                employment_start="2021",
                employment_end="2024",
                responsibilities=[
                    "Engineered asynchronous data processing pipelines using Python and PostgreSQL.",
                    "Improved API response times through caching and SQL query optimization.",
                ],
                technologies=["Python", "PostgreSQL", "Flask", "Docker"],
                original_text="Backend Software Engineer @ Acme Cloud Systems (2021-2024)",
            )
        ],
        projects=[
            ProjectRecord(
                project_name="Distributed Task Scheduler",
                description="Built a distributed task coordinator using Python and Redis.",
                technologies=["Python", "Redis"],
                project_url="https://github.com/alice/scheduler",
                original_text="Distributed Task Scheduler: Built with Python and Redis.",
            )
        ],
    )


# =============================================================================
# 1. Evidence Reference Resolver Tests
# =============================================================================

class TestEvidenceReferenceResolver:
    """Verifies that evidence IDs and field paths resolve accurately against ResumeProfile."""

    def test_valid_experience_reference_resolves(self, sample_candidate_profile):
        resolver = EvidenceReferenceResolver()
        is_resolved, err, val = resolver.resolve_reference(
            entity_id="exp_0",
            field_path="experience[0].responsibilities[0]",
            profile=sample_candidate_profile,
        )
        assert is_resolved is True
        assert err is None
        assert "asynchronous data processing pipelines" in val

    def test_valid_project_reference_resolves(self, sample_candidate_profile):
        resolver = EvidenceReferenceResolver()
        is_resolved, err, val = resolver.resolve_reference(
            entity_id="proj_0",
            field_path="projects[0].description",
            profile=sample_candidate_profile,
        )
        assert is_resolved is True
        assert err is None
        assert "distributed task coordinator" in val

    def test_invalid_experience_index_out_of_bounds_rejected(self, sample_candidate_profile):
        resolver = EvidenceReferenceResolver()
        is_resolved, err, _ = resolver.resolve_reference(
            entity_id="exp_5",
            field_path="experience[5].responsibilities[0]",
            profile=sample_candidate_profile,
        )
        assert is_resolved is False
        assert "out of bounds" in err

    def test_invalid_responsibility_index_out_of_bounds_rejected(self, sample_candidate_profile):
        resolver = EvidenceReferenceResolver()
        is_resolved, err, _ = resolver.resolve_reference(
            entity_id="exp_0",
            field_path="experience[0].responsibilities[99]",
            profile=sample_candidate_profile,
        )
        assert is_resolved is False
        assert "Responsibility index 99 out of bounds" in err

    def test_unknown_entity_id_prefix_rejected(self, sample_candidate_profile):
        resolver = EvidenceReferenceResolver()
        is_resolved, err, _ = resolver.resolve_reference(
            entity_id="unsupported_target",
            field_path="some_field",
            profile=sample_candidate_profile,
        )
        assert is_resolved is False
        assert "Unrecognized entity identifier prefix" in err

    def test_clarification_tagged_as_resume_extracted_rejected(self):
        resolver = EvidenceReferenceResolver()
        viol = resolver.validate_provenance_tier(
            tier=ProvenanceTier.RESUME_EXTRACTED,
            source_path="user_clarifications.docker_exp",
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.MISSING_EVIDENCE_REFERENCE
        assert "User clarification cannot be tagged with RESUME_EXTRACTED" in viol.message


# =============================================================================
# 2. Candidate Skill Grounding Validator Tests
# =============================================================================

class TestCandidateSkillGroundingValidator:
    """Verifies that candidate assertions require whitelist evidence, while recommendations do not."""

    def test_evidenced_skill_assertion_passes(self, sample_candidate_profile):
        validator = CandidateSkillGroundingValidator()
        viol = validator.validate_skill_claim(
            skill_name="Python",
            context=ClaimContext.CANDIDATE_ASSERTION,
            profile=sample_candidate_profile,
        )
        assert viol is None

    def test_synonym_skill_assertion_passes(self, sample_candidate_profile):
        validator = CandidateSkillGroundingValidator()
        # "Postgres" maps to "PostgreSQL" via SYNONYM_LOOKUP
        viol = validator.validate_skill_claim(
            skill_name="Postgres",
            context=ClaimContext.CANDIDATE_ASSERTION,
            profile=sample_candidate_profile,
        )
        assert viol is None

    def test_unsupported_candidate_skill_assertion_rejected(self, sample_candidate_profile):
        validator = CandidateSkillGroundingValidator()
        # Candidate has Flask, but never evidenced FastAPI
        viol = validator.validate_skill_claim(
            skill_name="FastAPI",
            context=ClaimContext.CANDIDATE_ASSERTION,
            profile=sample_candidate_profile,
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.UNGROUNDED_SKILL_CLAIM
        assert "FastAPI" in viol.message

    def test_future_learning_recommendation_citing_missing_skill_passes(self, sample_candidate_profile):
        validator = CandidateSkillGroundingValidator()
        # In advisory recommendations (milestones/projects), missing skills are legitimately recommended
        viol = validator.validate_skill_claim(
            skill_name="FastAPI",
            context=ClaimContext.ADVISORY_RECOMMENDATION,
            profile=sample_candidate_profile,
        )
        assert viol is None

    def test_unsupported_technology_kubernetes_in_resume_bullet_rejected(self, sample_candidate_profile):
        validator = CandidateSkillGroundingValidator()
        viol = validator.validate_skill_claim(
            skill_name="Kubernetes",
            context=ClaimContext.CANDIDATE_ASSERTION,
            profile=sample_candidate_profile,
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.UNGROUNDED_SKILL_CLAIM


# =============================================================================
# 3. Entity & Tenure Validator Tests
# =============================================================================

class TestEntityTenureValidator:
    """Verifies that employment metadata (titles, companies, dates) is protected against fabrication."""

    def test_exact_job_title_and_company_passes(self, sample_candidate_profile):
        validator = EntityTenureValidator()
        source_exp = sample_candidate_profile.experience[0]
        violations = validator.validate_work_experience_metadata(
            source_exp=source_exp,
            claimed_title="Backend Software Engineer",
            claimed_company="Acme Cloud Systems",
        )
        assert len(violations) == 0

    def test_fabricated_job_title_rejected(self, sample_candidate_profile):
        validator = EntityTenureValidator()
        source_exp = sample_candidate_profile.experience[0]
        violations = validator.validate_work_experience_metadata(
            source_exp=source_exp,
            claimed_title="Chief Technology Officer",
            claimed_company="Acme Cloud Systems",
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.TITLE_ALTERATION_CLAIM
        assert "Chief Technology Officer" in violations[0].message

    def test_fabricated_organization_name_rejected(self, sample_candidate_profile):
        validator = EntityTenureValidator()
        source_exp = sample_candidate_profile.experience[0]
        violations = validator.validate_work_experience_metadata(
            source_exp=source_exp,
            claimed_title="Backend Software Engineer",
            claimed_company="Google LLC",
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.ORGANIZATION_ALTERATION_CLAIM
        assert "Google LLC" in violations[0].message

    def test_project_record_represented_as_employment_tenure_rejected(self, sample_candidate_profile):
        validator = EntityTenureValidator()
        source_proj = sample_candidate_profile.projects[0]
        viol = validator.validate_no_project_as_employment(
            source_record=source_proj,
            claimed_as_employment=True,
        )
        assert viol is not None
        assert viol.violation_type == ValidationViolationType.TENURE_OVERSTATEMENT_CLAIM


# =============================================================================
# 4. Metric Grounding Validator Tests
# =============================================================================

class TestMetricGroundingValidator:
    """Verifies that quantitative claims require disclosure and rejects false verification."""

    def test_metric_with_self_reported_disclosure_passes(self):
        validator = MetricGroundingValidator()
        violations = validator.validate_metric_claim(
            text="Reduced API query latency by 45% through database indexing.",
            disclosure_status=MetricDisclosureStatus.SELF_REPORTED_METRIC,
        )
        assert len(violations) == 0

    def test_metric_without_disclosure_rejected(self):
        validator = MetricGroundingValidator()
        violations = validator.validate_metric_claim(
            text="Increased system throughput by 30% and cut costs by $20,000.",
            disclosure_status=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.UNDISCLOSED_METRIC_CLAIM

    def test_false_independent_verification_statement_rejected(self):
        validator = MetricGroundingValidator()
        violations = validator.validate_metric_claim(
            text="Independently verified in production benchmark that Redis caching reduced latency.",
            disclosure_status=MetricDisclosureStatus.SELF_REPORTED_METRIC,
        )
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.FALSE_INDEPENDENT_VERIFICATION_CLAIM


# =============================================================================
# 5. PII Leakage Validator Tests
# =============================================================================

class TestPIILeakageValidator:
    """Verifies that candidate personal contact details are caught without echoing raw PII."""

    def test_email_leakage_detected(self):
        validator = PIILeakageValidator()
        violations = validator.validate_text("Contact me at alice.smith@example.com for details.")
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.PII_LEAKAGE
        # Raw email must NOT be stored in invalid_value
        assert violations[0].invalid_value == "[REDACTED_EMAIL]"

    def test_phone_number_leakage_detected(self):
        validator = PIILeakageValidator()
        violations = validator.validate_text("Available at +1 555-019-2834 on weekdays.")
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.PII_LEAKAGE
        assert violations[0].invalid_value == "[REDACTED_PHONE]"

    def test_street_address_leakage_detected(self):
        validator = PIILeakageValidator()
        violations = validator.validate_text("Located at 742 Evergreen Street near the tech hub.")
        assert len(violations) == 1
        assert violations[0].violation_type == ValidationViolationType.PII_LEAKAGE
        assert violations[0].invalid_value == "[REDACTED_ADDRESS]"

    def test_clean_technical_text_passes(self):
        validator = PIILeakageValidator()
        violations = validator.validate_text(
            "Architected asynchronous ingestion pipeline using Python, Redis, and PostgreSQL."
        )
        assert len(violations) == 0


# =============================================================================
# 6. Consolidated Deterministic Workflow Validator Tests
# =============================================================================

class TestConsolidatedDeterministicWorkflowValidator:
    """Tests end-to-end deliverable validation across complete report fixtures."""

    def test_valid_resume_optimization_report_passes(self, sample_candidate_profile):
        validator = DeterministicWorkflowValidator()
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Strong candidate alignment in Python and PostgreSQL backend engineering.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b1",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Engineered high-throughput asynchronous pipelines with Python and PostgreSQL.",
                    highlighted_skills=["Python", "PostgreSQL"],
                    alignment_rationale="Emphasized verified database and programming proficiencies.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                ),
                OptimizedBulletPoint(
                    bullet_id="b2",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[1]",
                    improved_star_bullet="Accelerated database query execution by 35% through indexing and caching.",
                    highlighted_skills=["PostgreSQL"],
                    alignment_rationale="Quantified latency reduction.",
                    metric_disclosure=MetricDisclosureStatus.SELF_REPORTED_METRIC,
                ),
            ],
            section_reordering_recommendations=["Elevate backend experience"],
            ats_keyword_density_feedback=["Solid Python keyword coverage"],
        )
        res = validator.validate_resume_optimization(report, sample_candidate_profile)
        assert res.is_valid is True
        assert len(res.violations) == 0

    def test_invalid_resume_optimization_with_ungrounded_skill_and_pii_rejected(self, sample_candidate_profile):
        validator = DeterministicWorkflowValidator()
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Contact candidate at candidate@domain.com for interview.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b1",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Built machine learning models using PyTorch and Kubernetes.",
                    highlighted_skills=["PyTorch", "Kubernetes"],  # Neither is in candidate profile!
                    alignment_rationale="Added unevidenced skills.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                ),
            ],
        )
        res = validator.validate_resume_optimization(report, sample_candidate_profile)
        assert res.is_valid is False
        violation_types = [v.violation_type for v in res.violations]
        assert ValidationViolationType.PII_LEAKAGE in violation_types
        assert ValidationViolationType.UNGROUNDED_SKILL_CLAIM in violation_types

    def test_valid_remediation_plan_referencing_missing_jd_skills_passes(self, sample_candidate_profile):
        validator = DeterministicWorkflowValidator()
        plan = CareerRemediationPlan(
            executive_summary="30-day preparation roadmap to bridge Kubernetes and FastAPI gaps.",
            milestones=[
                UpskillingMilestone(
                    timeframe="Days 1-30",
                    target_missing_skills=["FastAPI", "Kubernetes"],  # Allowed in advisory context!
                    learning_objectives=["Master FastAPI dependency injection", "Deploy manifests to Minikube"],
                    concrete_deliverable="Containerized service running on local Kubernetes cluster",
                )
            ],
            portfolio_projects=[
                PortfolioProjectSpecification(
                    project_concept="Microservices Orchestrator",
                    architecture_overview="FastAPI gateway with gRPC services deployed on Kubernetes",
                    targeted_jd_gaps=["Kubernetes orchestration"],
                    recommended_tech_stack=["FastAPI", "Kubernetes", "Docker"],
                    production_grade_checklist=["Helm chart", "Readiness probes"],
                )
            ],
            estimated_readiness_timeline="4 weeks",
        )
        res = validator.validate_remediation_plan(plan, sample_candidate_profile)
        assert res.is_valid is True
        assert len(res.violations) == 0
