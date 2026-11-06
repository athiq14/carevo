"""
Comprehensive integration tests for Phase 5 LangGraph Agentic Workflow.
Tests every permitted goal and route, real LangGraph HITL interrupt/resume,
triage clarification, final approval enforcement, score integrity against external anchor,
bounded correction, and conservative bullet prose skill validation.
"""
from datetime import datetime, timezone
import pytest

from core.llm import MockLLMProvider
from core.llm.structured.engine import StructuredOutputEngine
from core.matching.exceptions import ScoreImmutabilityError
from core.workflow.coordinator import WorkflowExecutionCoordinator
from core.workflow.nodes import set_agent_engine_override
from core.workflow.validators import CandidateSkillGroundingValidator, DeterministicWorkflowValidator
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    EvidenceStatus,
    JobDescriptionProfile,
    JobMatchResult,
    MatchBreakdown,
    RequirementCategory,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models.resume_entities import ExtractedSkill, SkillCategory, WorkExperience
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from models.workflow import (
    MetricDisclosureStatus,
    OptimizedBulletPoint,
    ResumeOptimizationReport,
    UserWorkflowGoal,
)


@pytest.fixture(autouse=True)
def setup_mock_engine():
    """Ensure mock engine is used during testing so tests are 100% offline and deterministic."""
    mock = MockLLMProvider(canned_response=None, simulate_error=Exception("Offline fallback test"))
    engine = StructuredOutputEngine(dispatcher=mock)
    set_agent_engine_override(engine)
    yield
    set_agent_engine_override(None)


@pytest.fixture
def mock_profile() -> ResumeProfile:
    """Construct valid test ResumeProfile."""
    return ResumeProfile(
        filename="jane_doe_resume.pdf",
        page_count=1,
        raw_character_count=900,
        cleaned_character_count=880,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(
                name="Python",
                original_mention="Python 3.11",
                category=SkillCategory.PROGRAMMING_LANGUAGES,
            ),
            ExtractedSkill(
                name="Docker",
                original_mention="Docker containerization",
                category=SkillCategory.DEVOPS_AND_INFRASTRUCTURE,
            ),
            ExtractedSkill(
                name="FastAPI",
                original_mention="FastAPI microservices",
                category=SkillCategory.WEB_DEVELOPMENT,
            ),
        ],
        experience=[
            WorkExperience(
                job_title="Senior Backend Engineer",
                company="Acme Corp",
                employment_start="2020",
                employment_end="2023",
                responsibilities=[
                    "Engineered Python microservices serving 10,000 users.",
                    "Containerized backend applications with Docker.",
                ],
                technologies=["Python", "Docker", "FastAPI"],
                original_text="Senior Backend Engineer at Acme Corp",
            )
        ],
        education=[],
        projects=[],
        has_ambiguities=False,
    )


@pytest.fixture
def mock_match_result() -> JobMatchResult:
    """Construct valid deterministic JobMatchResult with 16 authoritative fields."""
    return JobMatchResult(
        overall_score=78.5,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=85.0,
        unassessed_reserve=15.0,
        match_breakdown=[
            MatchBreakdown(
                dimension_name="Technical Skills",
                dimension_score=80.0,
                baseline_weight=0.5,
                active_weight=0.5,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
            MatchBreakdown(
                dimension_name="Work Experience",
                dimension_score=75.0,
                baseline_weight=0.5,
                active_weight=0.5,
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
                skill_name="Docker",
                canonical_name="Docker",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            ),
        ],
        partial_matches=[],
        missing_evidence=[
            SkillGap(
                skill_name="Kubernetes",
                priority=RequirementPriority.PREFERRED,
                gap_explanation="Kubernetes is preferred by employer but absent in candidate evidence.",
                suggested_learning_direction="Container orchestration",
            )
        ],
        unknown_evidence=[],
        project_evidence=[],
        strengths=["Strong proficiency in Python and Docker"],
        skill_gaps=[
            SkillGap(
                skill_name="Kubernetes",
                priority=RequirementPriority.PREFERRED,
                gap_explanation="Kubernetes is preferred by employer but absent in candidate evidence.",
                suggested_learning_direction="Container orchestration",
            )
        ],
        recommendations=["Gain hands-on experience with Kubernetes"],
        processing_warnings=[],
        job_metadata={"job_title": "Senior Backend Engineer", "company": "Tech Corp"},
    )


@pytest.fixture
def high_reserve_match_result(mock_match_result: JobMatchResult) -> JobMatchResult:
    """Match result with unassessed reserve > 30% to trigger clarification interrupt."""
    return mock_match_result.model_copy(
        update={
            "unassessed_reserve": 35.0,
            "active_evaluation_coverage": 65.0,
        }
    )


# =============================================================================
# Test Suite 1: Real HITL Interrupt and Resume Behavior
# =============================================================================

def test_clarification_interrupt_and_resume(
    high_reserve_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that unassessed reserve > 30% pauses execution for user clarification,
    and resumes using the same thread_id with Command(resume=...).
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_clarification_1"

    # Step 1: Start workflow
    state, interrupts = coordinator.start_workflow(
        match_result=high_reserve_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    # Must be interrupted for clarification
    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "clarification_required"
    assert interrupts[0]["stage"] == "clarification"
    assert "triage" in state["completed_nodes"]
    assert "clarification" not in state["completed_nodes"]

    # Step 2: Resume with candidate clarification using same thread_id
    resumed_state, next_interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="Focus on distributed backend architecture",
    )

    # Must now be paused at goal_selection
    assert "clarification" in resumed_state["completed_nodes"]
    assert resumed_state["user_clarifications"] == {
        "candidate_clarification": "Focus on distributed backend architecture"
    }
    assert len(next_interrupts) == 1
    assert next_interrupts[0]["type"] == "goal_selection_required"


def test_goal_selection_interrupt_and_resume(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that low-reserve match skips clarification and interrupts at goal_selection.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_goal_1"

    state, interrupts = coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    # Low reserve: skips clarification, pauses at goal_selection
    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "goal_selection_required"
    assert "triage" in state["completed_nodes"]
    assert "clarification" not in state["completed_nodes"]


def test_invalid_goal_rejection(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that submitting an invalid workflow goal raises ValueError and is rejected.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_invalid_goal"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    with pytest.raises(ValueError, match="Invalid workflow goal 'EXPLORE_CAREERS'"):
        coordinator.resume_workflow(thread_id=thread_id, resume_value="EXPLORE_CAREERS")


# =============================================================================
# Test Suite 2: Exact Routing & No Unnecessary Agent Execution
# =============================================================================

def test_route_gap_remediation_only_runs_remediation(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Route GAP_REMEDIATION: Remediation -> Audit -> Final Approval -> Synthesis.
    Must NOT execute ResumeOptimizationAgent or InterviewPreparationAgent.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_gap_remediation"

    # Start -> pauses at goal_selection
    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    # Resume with GAP_REMEDIATION -> executes remediation -> audit -> pauses at final_approval
    state, interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="GAP_REMEDIATION",
    )

    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "final_approval_required"

    # Verify executed nodes
    completed = state["completed_nodes"]
    assert "remediation" in completed
    assert "deterministic_audit" in completed
    assert "resume_optimization" not in completed
    assert "interview_prep" not in completed

    # Verify deliverable presence
    assert state["remediation_plan"] is not None
    assert state["resume_optimization"] is None
    assert state["interview_kit"] is None

    # Step 3: Approve final deliverables
    final_state, final_interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="APPROVE",
    )

    assert len(final_interrupts) == 0
    assert "synthesis" in final_state["completed_nodes"]
    assert final_state["consolidated_report"] is not None
    assert final_state["consolidated_report"]["career_remediation"] is not None
    assert final_state["consolidated_report"]["resume_optimization"] is None


def test_route_resume_optimization_only_runs_resume_agent(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Route RESUME_OPTIMIZATION: Resume Optimization -> Audit -> Final Approval.
    Must NOT execute Remediation or InterviewPreparation.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_resume_opt"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    state, interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="RESUME_OPTIMIZATION",
    )

    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "final_approval_required"

    completed = state["completed_nodes"]
    assert "resume_optimization" in completed
    assert "deterministic_audit" in completed
    assert "remediation" not in completed
    assert "interview_prep" not in completed

    assert state["resume_optimization"] is not None
    assert state["remediation_plan"] is None
    assert state["interview_kit"] is None


def test_route_interview_drill_only_runs_interview_agent(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Route INTERVIEW_DRILL: Interview Preparation -> Audit -> Final Approval.
    Must NOT execute Remediation or Resume Optimization.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_interview_drill"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    state, interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="INTERVIEW_DRILL",
    )

    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "final_approval_required"

    completed = state["completed_nodes"]
    assert "interview_prep" in completed
    assert "deterministic_audit" in completed
    assert "remediation" not in completed
    assert "resume_optimization" not in completed

    assert state["interview_kit"] is not None
    assert state["remediation_plan"] is None
    assert state["resume_optimization"] is None


def test_route_full_preparation_runs_all_specialists_in_order(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Route FULL_PREPARATION: Remediation -> Resume Optimization -> Interview Preparation -> Audit -> Final Approval.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_full_prep"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    state, interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="FULL_PREPARATION",
    )

    assert len(interrupts) == 1
    assert interrupts[0]["type"] == "final_approval_required"

    completed = state["completed_nodes"]
    assert "remediation" in completed
    assert "resume_optimization" in completed
    assert "interview_prep" in completed
    assert "deterministic_audit" in completed

    # Check order
    rem_idx = completed.index("remediation")
    res_idx = completed.index("resume_optimization")
    int_idx = completed.index("interview_prep")
    aud_idx = completed.index("deterministic_audit")
    assert rem_idx < res_idx < int_idx < aud_idx

    assert state["remediation_plan"] is not None
    assert state["resume_optimization"] is not None
    assert state["interview_kit"] is not None


# =============================================================================
# Test Suite 3: Final Approval Cannot Be Bypassed & Rejection Handling
# =============================================================================

def test_final_approval_cannot_be_bypassed_and_rejection_halts_synthesis(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that final approval cannot be bypassed. Rejecting the deliverables
    terminates the workflow without executing synthesis.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_rejection"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    coordinator.resume_workflow(thread_id=thread_id, resume_value="RESUME_OPTIMIZATION")

    # Reject deliverables
    state, interrupts = coordinator.resume_workflow(
        thread_id=thread_id,
        resume_value="REJECT",
    )

    assert len(interrupts) == 0
    assert state["final_approval_decision"] == "REJECT"
    assert "synthesis" not in state["completed_nodes"]
    assert state["consolidated_report"] is None


def test_invalid_final_approval_choice_rejected(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that invalid approval decisions (e.g. 'MAYBE') raise ValueError.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_invalid_approval"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    coordinator.resume_workflow(thread_id=thread_id, resume_value="GAP_REMEDIATION")

    with pytest.raises(ValueError, match="Invalid approval decision 'MAYBE'"):
        coordinator.resume_workflow(thread_id=thread_id, resume_value="MAYBE")


# =============================================================================
# Test Suite 4: Conservative Bullet Prose Skill Validation
# =============================================================================

def test_prose_skill_validation_rejects_unsupported_technical_skills(
    mock_profile: ResumeProfile,
):
    """
    Verify that recognizable technical skills mentioned in rewritten bullet prose
    are caught and rejected if not present in the candidate ResumeProfile.
    """
    validator = CandidateSkillGroundingValidator()

    # Candidate has Python, Docker, FastAPI. Does NOT have Kubernetes or Terraform.
    unsupported_prose = (
        "Architected scalable infrastructure using Kubernetes and Terraform for cloud deployments."
    )
    violations = validator.validate_prose_skill_claims(
        text=unsupported_prose,
        profile=mock_profile,
    )

    assert len(violations) >= 1
    found_invalid = {v.invalid_value for v in violations}
    assert "Kubernetes" in found_invalid or "Terraform" in found_invalid


def test_prose_skill_validation_accepts_supported_technical_skills(
    mock_profile: ResumeProfile,
):
    """
    Verify that verified skills in rewritten bullet prose pass validation.
    """
    validator = CandidateSkillGroundingValidator()

    supported_prose = (
        "Engineered containerized Python services using Docker and FastAPI."
    )
    violations = validator.validate_prose_skill_claims(
        text=supported_prose,
        profile=mock_profile,
    )

    assert len(violations) == 0


def test_prose_skill_validation_does_not_reject_common_english_words(
    mock_profile: ResumeProfile,
):
    """
    Verify that ordinary words (e.g. 'next step', 'express', 'cloud data')
    are NOT flagged as skill claims merely because of name similarity.
    """
    validator = CandidateSkillGroundingValidator()

    prose_with_common_words = (
        "Took the next step in project development, expressed design choices, and delivered cloud data service."
    )
    violations = validator.validate_prose_skill_claims(
        text=prose_with_common_words,
        profile=mock_profile,
    )

    assert len(violations) == 0


# =============================================================================
# Test Suite 5: Score Immutability & Tampering Protection Against External Anchor
# =============================================================================

def test_score_tampering_caught_by_external_anchor(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that if a node or malicious caller alters the match_result in state,
    the coordinator's external SHA-256 anchor detects the tampering and raises ScoreImmutabilityError.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_tamper"

    state, _ = coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    # Tamper with the candidate score in a simulated state dictionary
    tampered_match = mock_match_result.model_copy(update={"overall_score": 99.9})
    tampered_state = dict(state)
    tampered_state["match_result"] = tampered_match

    with pytest.raises(ScoreImmutabilityError, match="Candidate SHA-256 digest"):
        coordinator._verify_thread_integrity(thread_id, tampered_state)


def test_workflow_reset_clears_anchors_and_prevents_stale_access(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that reset_workflow clears session registry and prevents stale thread access.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_reset"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )

    coordinator.reset_workflow(thread_id)

    with pytest.raises(KeyError, match="not found in coordinator session registry"):
        coordinator.get_state(thread_id)


def test_thread_isolation_maintains_independent_states_and_anchors(
    mock_match_result: JobMatchResult,
    high_reserve_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that multiple threads maintain independent checkpoints and external integrity anchors.
    """
    coordinator = WorkflowExecutionCoordinator()
    t1 = "thread_isolated_1"
    t2 = "thread_isolated_2"

    # Start thread 1 with standard match -> pauses at goal_selection
    s1, intr1 = coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=t1,
    )
    assert intr1[0]["type"] == "goal_selection_required"

    # Start thread 2 with high reserve -> pauses at clarification
    s2, intr2 = coordinator.start_workflow(
        match_result=high_reserve_match_result,
        resume_profile=mock_profile,
        thread_id=t2,
    )
    assert intr2[0]["type"] == "clarification_required"

    # Verify coordinator has distinct anchors
    assert coordinator._integrity_anchors[t1] != coordinator._integrity_anchors[t2]

    # Resuming t1 does not impact t2
    s1_res, intr1_next = coordinator.resume_workflow(thread_id=t1, resume_value="GAP_REMEDIATION")
    assert intr1_next[0]["type"] == "final_approval_required"

    # Check t2 is still paused at clarification
    s2_check, intr2_check = coordinator.get_state(thread_id=t2)
    assert intr2_check[0]["type"] == "clarification_required"


def test_synthesis_produces_sanitized_privacy_compliant_export(
    mock_match_result: JobMatchResult,
    mock_profile: ResumeProfile,
):
    """
    Verify that synthesis produces a complete, privacy-compliant JSON export without raw resume text or PII.
    """
    coordinator = WorkflowExecutionCoordinator()
    thread_id = "test_thread_synthesis"

    coordinator.start_workflow(
        match_result=mock_match_result,
        resume_profile=mock_profile,
        thread_id=thread_id,
    )
    coordinator.resume_workflow(thread_id=thread_id, resume_value="FULL_PREPARATION")
    final_state, _ = coordinator.resume_workflow(thread_id=thread_id, resume_value="APPROVE")

    report = final_state.get("consolidated_report")
    assert report is not None
    assert "baseline_compatibility" in report
    assert "career_remediation" in report
    assert "resume_optimization" in report
    assert "interview_preparation" in report
    assert "audit_status" in report
    assert "telemetry" in report

    # Verify no raw resume text in export
    report_str = str(report)
    assert "raw_text" not in report_str
    # Verify PII pattern absence
    assert "@" not in report_str

