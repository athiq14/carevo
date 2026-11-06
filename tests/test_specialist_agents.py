"""
Unit and integration tests for Phase 5.2 Specialist Agents:
TriageAgent, CareerRemediationAgent, ResumeOptimizationAgent, and InterviewPreparationAgent.
Verifies valid structured outputs, mock LLM execution, deterministic fallbacks,
factual grounding, metric disclosures, PII protection, and score immutability.
"""
import json
import pytest

from agents import (
    CareerRemediationAgent,
    InterviewPreparationAgent,
    ResumeOptimizationAgent,
    TriageAgent,
)
from core.llm.exceptions import LLMTimeoutError
from core.llm.mock_provider import MockLLMProvider
from core.llm.models import LLMResponse, LLMUsageMetadata
from core.llm.structured.engine import StructuredOutputEngine
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
    CareerRemediationPlan,
    InterviewPreparationKit,
    InterviewQuestionCategory,
    MetricDisclosureStatus,
    ResumeOptimizationReport,
    TriageAssessment,
    UserWorkflowGoal,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def candidate_profile() -> ResumeProfile:
    """Standardized candidate profile with verified Python/PostgreSQL experience."""
    return ResumeProfile(
        filename="john_doe_resume.pdf",
        page_count=1,
        raw_character_count=900,
        cleaned_character_count=880,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(name="Python", original_mention="Python 3.11", category=SkillCategory.PROGRAMMING_LANGUAGES),
            ExtractedSkill(name="PostgreSQL", original_mention="PostgreSQL database", category=SkillCategory.DATABASES),
            ExtractedSkill(name="Docker", original_mention="Docker", category=SkillCategory.DEVOPS_AND_INFRASTRUCTURE),
        ],
        experience=[
            WorkExperience(
                job_title="Backend Software Engineer",
                company="FinTech Solutions",
                employment_start="2021",
                employment_end="2024",
                responsibilities=[
                    "Engineered asynchronous processing services using Python.",
                    "Improved database query execution by 30% through indexing in PostgreSQL.",
                ],
                technologies=["Python", "PostgreSQL", "Docker"],
                original_text="Backend Software Engineer @ FinTech Solutions (2021-2024)",
            )
        ],
        projects=[
            ProjectRecord(
                project_name="Streaming Telemetry Ingestion",
                description="Built telemetry consumer service with Python.",
                technologies=["Python"],
                original_text="Streaming Telemetry Ingestion with Python",
            )
        ],
    )


@pytest.fixture
def match_result() -> JobMatchResult:
    """Authoritative JobMatchResult with 78% compatibility score and missing Kubernetes."""
    return JobMatchResult(
        overall_score=78.0,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=100.0,
        unassessed_reserve=0.0,
        match_breakdown=[
            MatchBreakdown(
                dimension_name="Technical Skills",
                dimension_score=80.0,
                baseline_weight=0.4,
                active_weight=0.4,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
            MatchBreakdown(
                dimension_name="Work Experience",
                dimension_score=85.0,
                baseline_weight=0.3,
                active_weight=0.3,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
            MatchBreakdown(
                dimension_name="Engineering Projects",
                dimension_score=70.0,
                baseline_weight=0.3,
                active_weight=0.3,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
        ],
        matched_skills=[
            SkillMatch(
                skill_name="Python",
                canonical_name="Python",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
            SkillMatch(
                skill_name="PostgreSQL",
                canonical_name="PostgreSQL",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
        ],
        missing_evidence=[
            SkillGap(
                skill_name="Kubernetes",
                priority=RequirementPriority.REQUIRED,
                gap_explanation="Kubernetes is required by employer but absent in resume.",
                suggested_learning_direction="Container orchestration and Helm charts",
            )
        ],
    )


@pytest.fixture
def jd_profile() -> JobDescriptionProfile:
    """Standardized JobDescriptionProfile."""
    return JobDescriptionProfile(
        original_description="Senior Backend Engineer needed with Python and Kubernetes.",
        job_title="Senior Backend Engineer",
        company="CloudScale Networks",
    )


def make_mock_engine(canned_json: str) -> StructuredOutputEngine:
    """Construct a StructuredOutputEngine powered by a MockLLMProvider returning canned_json."""
    mock_prov = MockLLMProvider(canned_response=canned_json)
    return StructuredOutputEngine(dispatcher=mock_prov)


# =============================================================================
# 1. Triage Agent Tests
# =============================================================================

class TestTriageAgent:
    """Tests evaluation diagnostics, goal suggestion rules, and deterministic fallback."""

    def test_triage_agent_mock_generation(self, match_result, jd_profile):
        canned_triage = json.dumps({
            "primary_bottleneck": "Missing container orchestration (Kubernetes) required for cloud deployment.",
            "key_strengths_summary": ["Verified Python asynchronous backend engineering", "PostgreSQL database optimization"],
            "critical_gaps_summary": ["Kubernetes container orchestration"],
            "recommended_focus_areas": ["Hands-on Kubernetes cluster administration"],
            "suggested_goal": "FULL_PREPARATION",
            "advisory_rationale": "Candidate is strong technically but requires Kubernetes upskilling.",
        })
        engine = make_mock_engine(canned_triage)
        agent = TriageAgent(engine=engine)

        assessment, telemetry = agent.run(match_result=match_result, jd_profile=jd_profile)

        assert isinstance(assessment, TriageAssessment)
        assert assessment.suggested_goal == UserWorkflowGoal.FULL_PREPARATION
        assert "Kubernetes" in assessment.critical_gaps_summary[0]
        assert telemetry["provider"] in {"mock", "mock_provider"}
        assert telemetry["fallback_mode"] is False

    def test_triage_agent_never_sets_user_goal(self, match_result):
        # Triage output must only provide suggested_goal; user_goal is reserved for human input
        assessment = TriageAgent.synthesize_deterministic_fallback(match_result)
        assert hasattr(assessment, "suggested_goal")
        assert not hasattr(assessment, "user_goal")

    def test_triage_deterministic_fallback_for_low_score(self, match_result):
        # Low score (< 60) should deterministically suggest GAP_REMEDIATION
        low_match = match_result.model_copy(update={"overall_score": 45.0}, deep=True)
        assessment = TriageAgent.synthesize_deterministic_fallback(low_match)
        assert assessment.suggested_goal == UserWorkflowGoal.GAP_REMEDIATION
        assert "45.0%" in assessment.advisory_rationale

    def test_triage_deterministic_fallback_for_high_score(self, match_result):
        # High score (>= 75) with minimal gaps should suggest INTERVIEW_DRILL
        high_match = match_result.model_copy(update={"overall_score": 88.0, "missing_evidence": []}, deep=True)
        assessment = TriageAgent.synthesize_deterministic_fallback(high_match)
        assert assessment.suggested_goal == UserWorkflowGoal.INTERVIEW_DRILL

    def test_triage_agent_graceful_fallback_on_provider_error(self, match_result):
        # When engine raises an error, agent returns clean fallback without raising
        faulty_dispatcher = MockLLMProvider(simulate_error=LLMTimeoutError("API Timeout"))
        agent = TriageAgent(dispatcher=faulty_dispatcher)

        assessment, telemetry = agent.run(match_result=match_result)
        assert isinstance(assessment, TriageAssessment)
        assert telemetry["fallback_mode"] is True
        assert "API Timeout" in telemetry["fallback_reason"]


# =============================================================================
# 2. Career Remediation Agent Tests
# =============================================================================

class TestCareerRemediationAgent:
    """Tests remediation planning, advisory skill citing, and validation integration."""

    def test_remediation_agent_mock_generation(self, candidate_profile, match_result, jd_profile):
        canned_plan = json.dumps({
            "executive_summary": "30-day strategic roadmap to bridge Kubernetes gap for backend role.",
            "milestones": [
                {
                    "timeframe": "Days 1-30",
                    "target_missing_skills": ["Kubernetes"],
                    "learning_objectives": ["Learn pod manifests", "Deploy services to Minikube"],
                    "concrete_deliverable": "Containerized service repo with Kubernetes manifests and CI pipeline.",
                }
            ],
            "portfolio_projects": [
                {
                    "project_concept": "Cloud-Native Task Coordinator",
                    "architecture_overview": "Python FastAPI services orchestrated on Kubernetes with PostgreSQL.",
                    "targeted_jd_gaps": ["Kubernetes"],
                    "recommended_tech_stack": ["Python", "Kubernetes", "PostgreSQL"],
                    "production_grade_checklist": ["90% test coverage", "Helm chart manifests"],
                }
            ],
            "estimated_readiness_timeline": "4 weeks",
        })
        engine = make_mock_engine(canned_plan)
        agent = CareerRemediationAgent(engine=engine)

        plan, telemetry = agent.run(match_result=match_result, profile=candidate_profile, jd_profile=jd_profile)

        assert isinstance(plan, CareerRemediationPlan)
        assert plan.milestones[0].target_missing_skills == ("Kubernetes",)
        assert telemetry["fallback_mode"] is False

    def test_remediation_agent_allows_missing_skills_in_advisory_context(self, candidate_profile, match_result):
        # Remediation plan legitimately references missing skill 'Kubernetes'
        fallback_plan = CareerRemediationAgent.synthesize_deterministic_fallback(match_result)
        assert "Kubernetes" in fallback_plan.milestones[0].target_missing_skills
        assert isinstance(fallback_plan.milestones, tuple)

    def test_remediation_agent_fallback_on_llm_failure(self, candidate_profile, match_result):
        faulty_dispatcher = MockLLMProvider(simulate_error=LLMTimeoutError("Connection timeout"))
        agent = CareerRemediationAgent(dispatcher=faulty_dispatcher)

        plan, telemetry = agent.run(match_result=match_result, profile=candidate_profile)
        assert isinstance(plan, CareerRemediationPlan)
        assert telemetry["fallback_mode"] is True
        assert len(plan.milestones) == 2


# =============================================================================
# 3. Resume Optimization Agent Tests
# =============================================================================

class TestResumeOptimizationAgent:
    """Tests STAR framing, factual evidence reference grounding, and metric disclosure."""

    def test_resume_optimization_mock_generation(self, candidate_profile, match_result, jd_profile):
        canned_report = json.dumps({
            "target_role_alignment_summary": "Candidate exhibits strong backend alignment in Python and PostgreSQL.",
            "optimized_bullets": [
                {
                    "bullet_id": "opt_1",
                    "source_entity_id": "exp_0",
                    "source_field_path": "experience[0].responsibilities[0]",
                    "improved_star_bullet": "Architected asynchronous processing microservices using Python.",
                    "highlighted_skills": ["Python"],
                    "alignment_rationale": "Framed verified Python responsibility with architectural action verb.",
                    "metric_disclosure": "NOT_APPLICABLE",
                },
                {
                    "bullet_id": "opt_2",
                    "source_entity_id": "exp_0",
                    "source_field_path": "experience[0].responsibilities[1]",
                    "improved_star_bullet": "Optimized database query latency by 30% through index re-engineering.",
                    "highlighted_skills": ["PostgreSQL"],
                    "alignment_rationale": "Quantified query latency optimization with metric disclosure.",
                    "metric_disclosure": "SELF_REPORTED_METRIC",
                },
            ],
            "section_reordering_recommendations": ["Highlight backend experience"],
            "ats_keyword_density_feedback": ["Strong Python presence"],
        })
        engine = make_mock_engine(canned_report)
        agent = ResumeOptimizationAgent(engine=engine)

        report, telemetry = agent.run(profile=candidate_profile, match_result=match_result, jd_profile=jd_profile)

        assert isinstance(report, ResumeOptimizationReport)
        assert len(report.optimized_bullets) == 2
        assert report.optimized_bullets[1].metric_disclosure == MetricDisclosureStatus.SELF_REPORTED_METRIC
        assert telemetry["fallback_mode"] is False

    def test_resume_agent_rejects_ungrounded_skill_and_falls_back(self, candidate_profile, match_result):
        # If LLM fabricates unevidenced skill 'Kubernetes' in candidate assertion, validator rejects it
        fabricated_skill_report = json.dumps({
            "target_role_alignment_summary": "Candidate profile alignment.",
            "optimized_bullets": [
                {
                    "bullet_id": "opt_1",
                    "source_entity_id": "exp_0",
                    "source_field_path": "experience[0].responsibilities[0]",
                    "improved_star_bullet": "Engineered Kubernetes clusters and orchestrated microservices.",
                    "highlighted_skills": ["Kubernetes"],  # Fabricated! Candidate never had Kubernetes!
                    "alignment_rationale": "Fabricated claim.",
                    "metric_disclosure": "NOT_APPLICABLE",
                }
            ],
        })
        engine = make_mock_engine(fabricated_skill_report)
        agent = ResumeOptimizationAgent(engine=engine)

        report, telemetry = agent.run(profile=candidate_profile, match_result=match_result)
        # Validation caught the ungrounded claim -> safely triggered deterministic fallback
        assert telemetry["fallback_mode"] is True
        assert "UNGROUNDED_SKILL_CLAIM" in telemetry["fallback_reason"]
        # Fallback bullets strictly feature verified skills
        for bullet in report.optimized_bullets:
            for s in bullet.highlighted_skills:
                assert s in {"Python", "PostgreSQL", "Docker"}

    def test_resume_agent_rejects_pii_leakage_and_falls_back(self, candidate_profile, match_result):
        pii_leaking_report = json.dumps({
            "target_role_alignment_summary": "Contact candidate at secret.email@domain.com for interview.",
            "optimized_bullets": [
                {
                    "bullet_id": "opt_1",
                    "source_entity_id": "exp_0",
                    "source_field_path": "experience[0].responsibilities[0]",
                    "improved_star_bullet": "Built asynchronous microservices in Python.",
                    "highlighted_skills": ["Python"],
                    "alignment_rationale": "Standard bullet.",
                    "metric_disclosure": "NOT_APPLICABLE",
                }
            ],
        })
        engine = make_mock_engine(pii_leaking_report)
        agent = ResumeOptimizationAgent(engine=engine)

        report, telemetry = agent.run(profile=candidate_profile, match_result=match_result)
        assert telemetry["fallback_mode"] is True
        assert "PII_LEAKAGE" in telemetry["fallback_reason"]

    def test_resume_agent_deterministic_fallback_passes_all_checks(self, candidate_profile, match_result):
        fallback = ResumeOptimizationAgent.synthesize_deterministic_fallback(candidate_profile, match_result)
        assert isinstance(fallback, ResumeOptimizationReport)
        assert len(fallback.optimized_bullets) >= 1
        # Confirm that second bullet (which had 30% metric in fixture) has SELF_REPORTED_METRIC
        metric_bullets = [b for b in fallback.optimized_bullets if b.metric_disclosure == MetricDisclosureStatus.SELF_REPORTED_METRIC]
        assert len(metric_bullets) >= 1


# =============================================================================
# 4. Interview Preparation Agent Tests
# =============================================================================

class TestInterviewPreparationAgent:
    """Tests technical, gap-mitigation, and metric-challenge question formulation."""

    def test_interview_prep_agent_mock_generation(self, candidate_profile, match_result, jd_profile):
        canned_kit = json.dumps({
            "role_context_overview": "Interview preparation package for Senior Backend Engineer.",
            "technical_deep_dive_questions": [
                {
                    "question_id": "tech_1",
                    "category": "TECHNICAL",
                    "question_text": "How do you manage connection pooling and transactions in PostgreSQL?",
                    "interviewer_intent": "Assess database depth.",
                    "targeted_skill_or_gap": "PostgreSQL",
                    "suggested_talking_points": ["Discuss connection limits", "Mention indexing optimizations"],
                    "grounded_source_entity_id": "exp_0",
                }
            ],
            "gap_mitigation_questions": [
                {
                    "question_id": "gap_1",
                    "category": "GAP_PROBE",
                    "question_text": "How do you plan to transition your Docker container skills to Kubernetes?",
                    "interviewer_intent": "Assess container orchestration learning curve.",
                    "targeted_skill_or_gap": "Kubernetes",
                    "suggested_talking_points": ["Discuss Docker familiarity", "Cite learning roadmap"],
                    "grounded_source_entity_id": None,
                }
            ],
            "metric_challenge_questions": [
                {
                    "question_id": "metric_1",
                    "category": "METRIC_CHALLENGE",
                    "question_text": "What profiling tools were used to measure the 30% query improvement?",
                    "interviewer_intent": "Verify self-reported quantitative metric.",
                    "targeted_skill_or_gap": "PostgreSQL",
                    "suggested_talking_points": ["Cite EXPLAIN ANALYZE benchmarks"],
                    "grounded_source_entity_id": "exp_0",
                }
            ],
            "recommended_questions_for_interviewer": ["What is the production deployment cadence?"],
        })
        engine = make_mock_engine(canned_kit)
        agent = InterviewPreparationAgent(engine=engine)

        kit, telemetry = agent.run(match_result=match_result, profile=candidate_profile, jd_profile=jd_profile)

        assert isinstance(kit, InterviewPreparationKit)
        assert len(kit.technical_deep_dive_questions) == 1
        assert len(kit.gap_mitigation_questions) == 1
        assert len(kit.metric_challenge_questions) == 1
        assert telemetry["fallback_mode"] is False

    def test_interview_prep_deterministic_fallback(self, candidate_profile, match_result):
        fallback = InterviewPreparationAgent.synthesize_deterministic_fallback(match_result, candidate_profile)
        assert isinstance(fallback, InterviewPreparationKit)
        assert len(fallback.technical_deep_dive_questions) >= 1
        assert len(fallback.gap_mitigation_questions) >= 1
        assert len(fallback.metric_challenge_questions) >= 1
        assert len(fallback.recommended_questions_for_interviewer) >= 2

    def test_interview_prep_fallback_on_provider_error(self, candidate_profile, match_result):
        faulty_dispatcher = MockLLMProvider(simulate_error=LLMTimeoutError("Service unavailable"))
        agent = InterviewPreparationAgent(dispatcher=faulty_dispatcher)

        kit, telemetry = agent.run(match_result=match_result, profile=candidate_profile)
        assert isinstance(kit, InterviewPreparationKit)
        assert telemetry["fallback_mode"] is True
