"""
Unit and integration tests for LangGraph TypedDict state, reducers,
and immutable Pydantic V2 workflow models (Phase 5.1).
"""
from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.workflow.state import (
    CareerWorkflowState,
    reduce_clarifications,
    reduce_provenance_records,
    reduce_token_usage,
    reduce_trace_events,
    reduce_unique_strings,
    reduce_validation_reports,
    reduce_workflow_errors,
)
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    JobDescriptionProfile,
    JobMatchResult,
)
from models.resume_entities import ExtractedSkill, SkillCategory
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from models.workflow import (
    CareerRemediationPlan,
    ClaimContext,
    DeterministicValidationReport,
    DeterministicValidationViolation,
    EvidenceProvenanceRecord,
    InterviewPreparationKit,
    InterviewQuestion,
    InterviewQuestionCategory,
    MetricDisclosureStatus,
    OptimizedBulletPoint,
    PortfolioProjectSpecification,
    ProvenanceTier,
    ResumeOptimizationReport,
    TriageAssessment,
    URLVerificationStatus,
    UpskillingMilestone,
    UserWorkflowGoal,
    ValidationViolationType,
    WorkflowTraceEvent,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

def make_dummy_profiles() -> tuple[ResumeProfile, JobDescriptionProfile, JobMatchResult]:
    """Construct minimal valid Phase 1-4 objects for state testing."""
    resume_prof = ResumeProfile(
        filename="alice_resume.pdf",
        page_count=1,
        raw_character_count=500,
        cleaned_character_count=480,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES)],
    )
    jd_prof = JobDescriptionProfile(
        original_description="Senior Python Developer needed.",
        job_title="Senior Python Developer",
        company="Acme Corp",
    )
    match_res = JobMatchResult(
        overall_score=85.0,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=100.0,
        unassessed_reserve=0.0,
    )
    return resume_prof, jd_prof, match_res


# =============================================================================
# 1. State Reducer Unit Tests
# =============================================================================

class TestWorkflowStateReducers:
    """Verifies that all pure reducer functions accumulate properly without duplicates."""

    def test_reduce_unique_strings(self):
        existing = ["triage", "remediation"]
        update = ["remediation", "resume_opt", "audit"]
        merged = reduce_unique_strings(existing, update)
        assert merged == ["triage", "remediation", "resume_opt", "audit"]

    def test_reduce_unique_strings_handles_empty(self):
        assert reduce_unique_strings([], ["a", "b", "a"]) == ["a", "b"]
        assert reduce_unique_strings(["a"], []) == ["a"]

    def test_reduce_trace_events(self):
        evt1 = WorkflowTraceEvent(event_id="e1", node_name="n1", event_type="START", message="Started")
        evt2 = WorkflowTraceEvent(event_id="e2", node_name="n1", event_type="END", message="Ended")
        evt1_dup = WorkflowTraceEvent(event_id="e1", node_name="n1", event_type="START", message="Dup")

        res = reduce_trace_events([evt1], [evt1_dup, evt2])
        assert len(res) == 2
        assert [e.event_id for e in res] == ["e1", "e2"]

    def test_reduce_validation_reports(self):
        r1 = DeterministicValidationReport(report_id="r1", is_valid=True, target_deliverable_type="Test1")
        r2 = DeterministicValidationReport(report_id="r2", is_valid=False, target_deliverable_type="Test2")
        r1_dup = DeterministicValidationReport(report_id="r1", is_valid=True, target_deliverable_type="Test1")

        res = reduce_validation_reports([r1], [r1_dup, r2])
        assert len(res) == 2
        assert [r.report_id for r in res] == ["r1", "r2"]

    def test_reduce_provenance_records(self):
        p1 = EvidenceProvenanceRecord(
            provenance_id="p1", target_entity_id="exp_0", source_field_path="exp[0]", provenance_tier=ProvenanceTier.RESUME_EXTRACTED
        )
        p2 = EvidenceProvenanceRecord(
            provenance_id="p2", target_entity_id="proj_0", source_field_path="proj[0]", provenance_tier=ProvenanceTier.USER_CONFIRMED
        )
        p1_dup = EvidenceProvenanceRecord(
            provenance_id="p1", target_entity_id="exp_0", source_field_path="exp[0]", provenance_tier=ProvenanceTier.RESUME_EXTRACTED
        )

        res = reduce_provenance_records([p1], [p1_dup, p2])
        assert len(res) == 2
        assert [p.provenance_id for p in res] == ["p1", "p2"]

    def test_reduce_token_usage(self):
        existing = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        update = {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        res = reduce_token_usage(existing, update)
        assert res["prompt_tokens"] == 300
        assert res["completion_tokens"] == 130
        assert res["total_tokens"] == 430

    def test_reduce_workflow_errors(self):
        err1 = {"node": "remediation", "error": "rate limit"}
        err2 = {"node": "interview", "error": "timeout"}
        res = reduce_workflow_errors([err1], [err1, err2])
        assert len(res) == 2
        assert res == [err1, err2]

    def test_reduce_clarifications(self):
        existing = {"docker_years": "2 years"}
        update = {"github_url": "https://github.com/alice/project"}
        res = reduce_clarifications(existing, update)
        assert res == {"docker_years": "2 years", "github_url": "https://github.com/alice/project"}


# =============================================================================
# 2. Immutable Pydantic Deliverables Tests
# =============================================================================

class TestImmutableDeliverableModels:
    """Verifies that all Phase 5 deliverable schemas enforce immutability and valid serialization."""

    def test_triage_assessment_immutability(self):
        triage = TriageAssessment(
            primary_bottleneck="Missing cloud deployment experience",
            key_strengths_summary=["Expert Python", "PostgreSQL database design"],
            critical_gaps_summary=["AWS infrastructure"],
            recommended_focus_areas=["Containerization & ECS"],
            suggested_goal=UserWorkflowGoal.GAP_REMEDIATION,
            advisory_rationale="Prioritize cloud infrastructure upskilling before interview.",
        )
        with pytest.raises(ValidationError):
            # Attempt mutation on frozen model
            triage.primary_bottleneck = "Mutated"

    def test_remediation_plan_json_serialization(self):
        plan = CareerRemediationPlan(
            executive_summary="Targeted 30-day upskilling roadmap for backend role.",
            milestones=[
                UpskillingMilestone(
                    timeframe="Days 1-30",
                    target_missing_skills=["FastAPI", "Docker"],
                    learning_objectives=["Build REST APIs", "Write Dockerfiles"],
                    concrete_deliverable="Containerized REST API repo",
                )
            ],
            portfolio_projects=[
                PortfolioProjectSpecification(
                    project_concept="Async Task Worker",
                    architecture_overview="FastAPI producer + Celery workers + Redis",
                    targeted_jd_gaps=["Asynchronous architectures"],
                    recommended_tech_stack=["FastAPI", "Redis", "Docker"],
                    production_grade_checklist=["90%+ unit test coverage", "CI pipeline"],
                )
            ],
            estimated_readiness_timeline="4 weeks",
        )
        json_str = plan.model_dump_json()
        restored = CareerRemediationPlan.model_validate_json(json_str)
        assert restored == plan
        assert restored.milestones[0].target_missing_skills == ("FastAPI", "Docker")

    def test_optimized_bullet_point_immutability(self):
        bullet = OptimizedBulletPoint(
            bullet_id="b1",
            source_entity_id="exp_0",
            source_field_path="experience[0].responsibilities[0]",
            improved_star_bullet="Architected high-throughput data pipeline using Python.",
            highlighted_skills=["Python"],
            alignment_rationale="Emphasized architectural leadership.",
            metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        with pytest.raises(ValidationError):
            bullet.improved_star_bullet = "Hacked bullet"


# =============================================================================
# 3. LangGraph StateGraph Execution Integration
# =============================================================================

class TestLangGraphStateIntegration:
    """Verifies that official LangGraph compiles and executes CareerWorkflowState with partial updates."""

    def test_langgraph_partial_updates_and_reducers(self):
        resume_prof, jd_prof, match_res = make_dummy_profiles()

        # Define 2 test nodes returning partial state dictionaries
        def node_one(state: CareerWorkflowState) -> dict:
            return {
                "current_step": "step_one",
                "completed_nodes": ["node_one"],
                "token_usage_summary": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "suggested_goal": UserWorkflowGoal.FULL_PREPARATION,
            }

        def node_two(state: CareerWorkflowState) -> dict:
            return {
                "current_step": "step_two",
                "completed_nodes": ["node_two"],
                "token_usage_summary": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                "user_goal": UserWorkflowGoal.FULL_PREPARATION,
            }

        builder = StateGraph(CareerWorkflowState)
        builder.add_node("n1", node_one)
        builder.add_node("n2", node_two)
        builder.add_edge(START, "n1")
        builder.add_edge("n1", "n2")
        builder.add_edge("n2", END)

        graph = builder.compile(checkpointer=MemorySaver())

        initial_state: CareerWorkflowState = {
            "workflow_id": "wf-1234",
            "match_result": match_res,
            "resume_profile": resume_prof,
            "jd_profile": jd_prof,
        }

        thread_cfg = {"configurable": {"thread_id": "test-thread-1"}}
        final_state = graph.invoke(initial_state, config=thread_cfg)

        # Confirm authoritative fields preserved
        assert final_state["workflow_id"] == "wf-1234"
        assert final_state["match_result"].overall_score == 85.0

        # Confirm reducer accumulated completed nodes without duplicates
        assert final_state["completed_nodes"] == ["node_one", "node_two"]

        # Confirm reducer combined token usage
        assert final_state["token_usage_summary"]["prompt_tokens"] == 150
        assert final_state["token_usage_summary"]["completion_tokens"] == 75
        assert final_state["token_usage_summary"]["total_tokens"] == 225

        # Confirm partial scalar updates were set
        assert final_state["current_step"] == "step_two"
        assert final_state["suggested_goal"] == UserWorkflowGoal.FULL_PREPARATION
        assert final_state["user_goal"] == UserWorkflowGoal.FULL_PREPARATION
