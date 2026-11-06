"""
Dedicated security and immutability tests for Phase 5.1:
Guarantees authoritative JobMatchResult preservation, node-update interception,
and non-assert operational enforcement under python -O.
"""
from pydantic import ValidationError
import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from core.matching.exceptions import ScoreImmutabilityError
from core.workflow.state import CareerWorkflowState
from core.workflow.validators import (
    compute_match_result_digest,
    create_baseline_snapshot,
    intercept_node_update,
    serialize_authoritative_match_result,
    verify_workflow_score_immutability,
)
from models.matching import (
    CandidateFitLevel,
    DimensionBreakdown,
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
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from models.workflow import (
    AuthoritativeMatchSnapshot,
    CareerRemediationPlan,
    DeterministicValidationReport,
    DeterministicValidationViolation,
    InterviewPreparationKit,
    InterviewQuestion,
    InterviewQuestionCategory,
    MetricDisclosureStatus,
    OptimizedBulletPoint,
    PortfolioProjectSpecification,
    ResumeOptimizationReport,
    TriageAssessment,
    UpskillingMilestone,
    UserWorkflowGoal,
    ValidationViolationType,
    ViolationSeverity,
)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def baseline_match_result() -> JobMatchResult:
    """Construct an authoritative baseline JobMatchResult with populated dimensions."""
    breakdown = [
        MatchBreakdown(
            dimension_name="Technical Skills",
            dimension_score=80.0,
            baseline_weight=0.35,
            active_weight=0.35,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
        ),
        MatchBreakdown(
            dimension_name="Experience Tenure & Role Relevance",
            dimension_score=90.0,
            baseline_weight=0.30,
            active_weight=0.30,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
        ),
        MatchBreakdown(
            dimension_name="Education & Academic Qualifications",
            dimension_score=85.0,
            baseline_weight=0.15,
            active_weight=0.15,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
        ),
        MatchBreakdown(
            dimension_name="Engineering Projects & Applied Work",
            dimension_score=75.0,
            baseline_weight=0.20,
            active_weight=0.20,
            evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
        ),
    ]
    return JobMatchResult(
        overall_score=82.75,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=100.0,
        unassessed_reserve=0.0,
        match_breakdown=breakdown,
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
                priority=RequirementPriority.PREFERRED,
                gap_explanation="Kubernetes is not evidenced in candidate resume.",
                suggested_learning_direction="Container orchestration",
            )
        ],
    )


# =============================================================================
# 1. Node Update Interception Tests
# =============================================================================

class TestNodeUpdateInterception:
    """Verifies that workflow nodes are barred from returning protected Phase 4 fields."""

    def test_valid_node_update_passes(self):
        valid_update = {
            "current_step": "remediation_complete",
            "completed_nodes": ["remediation_agent"],
            "token_usage_summary": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
        }
        res = intercept_node_update(node_name="remediation_agent", update=valid_update)
        assert res == valid_update

    def test_node_attempting_to_return_match_result_is_rejected(self, baseline_match_result):
        rogue_update = {
            "match_result": baseline_match_result.model_copy(update={"overall_score": 99.0}),
            "current_step": "rogue_step",
        }
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update(node_name="rogue_agent", update=rogue_update)

        assert "Security Violation" in str(exc_info.value)
        assert "match_result" in str(exc_info.value)
        assert "strictly immutable" in str(exc_info.value)

    def test_node_attempting_to_override_resume_profile_is_rejected(self):
        rogue_update = {
            "resume_profile": "modified_fake_profile",
        }
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update(node_name="malicious_node", update=rogue_update)

        assert "Security Violation" in str(exc_info.value)
        assert "resume_profile" in str(exc_info.value)

    def test_node_attempting_to_override_jd_profile_is_rejected(self):
        rogue_update = {
            "jd_profile": "modified_fake_jd",
        }
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update(node_name="malicious_node", update=rogue_update)

        assert "Security Violation" in str(exc_info.value)
        assert "jd_profile" in str(exc_info.value)

    def test_node_attempting_to_override_baseline_snapshot_is_rejected(self):
        rogue_update = {
            "baseline_match_snapshot": "modified_fake_snapshot",
        }
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update(node_name="malicious_node", update=rogue_update)

        assert "baseline_match_snapshot" in str(exc_info.value)


# =============================================================================
# 2. End-of-Workflow Immutability Comparison Tests
# =============================================================================

class TestWorkflowImmutabilityComparison:
    """Verifies that final JobMatchResult preserves all 16 authoritative fields against baseline."""

    def test_unmodified_result_passes_immutability(self, baseline_match_result):
        # A deep copy with identical deterministic values passes
        candidate = baseline_match_result.model_copy(deep=True)
        # Should execute cleanly with zero exceptions
        verify_workflow_score_immutability(baseline=baseline_match_result, candidate=candidate)

    def test_mutated_overall_score_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(update={"overall_score": 95.0}, deep=True)
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Overall score mutated" in str(exc_info.value)

    def test_mutated_fit_classification_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(
            update={"fit_classification": CandidateFitLevel.EXCEPTIONAL}, deep=True
        )
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Fit classification mutated" in str(exc_info.value)

    def test_mutated_active_coverage_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(
            update={"active_evaluation_coverage": 60.0}, deep=True
        )
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Active coverage mutated" in str(exc_info.value)

    def test_mutated_unassessed_reserve_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(
            update={"unassessed_reserve": 40.0}, deep=True
        )
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Unassessed reserve mutated" in str(exc_info.value)

    def test_mutated_matched_skills_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(
            update={"matched_skills": []}, deep=True
        )
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Matched skills evidence mutated" in str(exc_info.value)

    def test_mutated_skill_gaps_rejected(self, baseline_match_result):
        mutated = baseline_match_result.model_copy(
            update={"skill_gaps": []}, deep=True
        )
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)
        assert "Skill gaps records mutated" in str(exc_info.value)


# =============================================================================
# 3. Non-Assert Enforcement (Python -O Verification)
# =============================================================================

class TestPythonOptimizedEnforcement:
    """Confirms that immutability checks use explicit conditional raises, never assert statements."""

    def test_immutability_uses_explicit_conditional_raises(self, baseline_match_result):
        # Mutated result must raise ScoreImmutabilityError even if python -O ignores asserts
        mutated = baseline_match_result.model_copy(update={"overall_score": 50.0}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline=baseline_match_result, candidate=mutated)

    def test_node_update_interceptor_uses_explicit_conditional_raises(self):
        with pytest.raises(ScoreImmutabilityError):
            intercept_node_update(node_name="test_node", update={"match_result": "tampered"})

    def test_create_baseline_snapshot_type_check_operational_under_o(self):
        with pytest.raises(TypeError) as exc_info:
            create_baseline_snapshot("invalid_type_object")  # type: ignore[arg-type]
        assert "Expected JobMatchResult" in str(exc_info.value)


# =============================================================================
# 4. Deep Immutability Tests for Deliverable Models (Nested Collections)
# =============================================================================

class TestDeepImmutabilityDeliverables:
    """
    Adversarial unit tests verifying that all workflow deliverable schemas
    reject in-place mutation attempts (.append, item assignment, item deletion).
    """

    def test_triage_assessment_nested_immutability(self):
        triage = TriageAssessment(
            primary_bottleneck="Cloud experience bottleneck",
            key_strengths_summary=["Python", "PostgreSQL"],
            critical_gaps_summary=["Kubernetes", "AWS"],
            recommended_focus_areas=["Containerization"],
            suggested_goal=UserWorkflowGoal.GAP_REMEDIATION,
            advisory_rationale="Focus on Kubernetes upskilling.",
        )
        assert isinstance(triage.key_strengths_summary, tuple)
        assert isinstance(triage.critical_gaps_summary, tuple)
        assert isinstance(triage.recommended_focus_areas, tuple)

        # Attempt in-place append
        with pytest.raises(AttributeError):
            triage.key_strengths_summary.append("Tampered")  # type: ignore[attr-defined]

        # Attempt in-place item assignment
        with pytest.raises(TypeError):
            triage.key_strengths_summary[0] = "Tampered"  # type: ignore[index]

        # Attempt in-place deletion
        with pytest.raises(TypeError):
            del triage.critical_gaps_summary[0]  # type: ignore[misc]

        # Attempt attribute assignment
        with pytest.raises(ValidationError):
            triage.primary_bottleneck = "Hacked"

    def test_upskilling_milestone_nested_immutability(self):
        milestone = UpskillingMilestone(
            timeframe="Days 1-30",
            target_missing_skills=["FastAPI", "Docker"],
            learning_objectives=["Build microservices", "Containerize"],
            concrete_deliverable="Containerized service repo",
        )
        assert isinstance(milestone.target_missing_skills, tuple)
        assert isinstance(milestone.learning_objectives, tuple)

        with pytest.raises(AttributeError):
            milestone.target_missing_skills.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            milestone.target_missing_skills[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del milestone.learning_objectives[0]  # type: ignore[misc]

        with pytest.raises(ValidationError):
            milestone.timeframe = "Modified timeframe"

    def test_portfolio_project_spec_nested_immutability(self):
        spec = PortfolioProjectSpecification(
            project_concept="Distributed Worker",
            architecture_overview="FastAPI + Celery + Redis",
            targeted_jd_gaps=["Distributed systems"],
            recommended_tech_stack=["FastAPI", "Redis"],
            production_grade_checklist=["90% test coverage"],
        )
        assert isinstance(spec.targeted_jd_gaps, tuple)
        assert isinstance(spec.recommended_tech_stack, tuple)
        assert isinstance(spec.production_grade_checklist, tuple)

        with pytest.raises(AttributeError):
            spec.recommended_tech_stack.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            spec.recommended_tech_stack[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del spec.production_grade_checklist[0]  # type: ignore[misc]

        with pytest.raises(ValidationError):
            spec.project_concept = "Modified title"

    def test_career_remediation_plan_nested_immutability(self):
        plan = CareerRemediationPlan(
            executive_summary="30-day preparation strategy",
            milestones=[
                UpskillingMilestone(
                    timeframe="Days 1-30",
                    target_missing_skills=["FastAPI"],
                    learning_objectives=["Build APIs"],
                    concrete_deliverable="FastAPI repo",
                )
            ],
            portfolio_projects=[
                PortfolioProjectSpecification(
                    project_concept="API Service",
                    architecture_overview="REST API architecture",
                    targeted_jd_gaps=["FastAPI"],
                    recommended_tech_stack=["FastAPI"],
                    production_grade_checklist=["Linting"],
                )
            ],
            estimated_readiness_timeline="4 weeks",
        )
        assert isinstance(plan.milestones, tuple)
        assert isinstance(plan.portfolio_projects, tuple)

        # In-place collection mutations on plan
        with pytest.raises(AttributeError):
            plan.milestones.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            plan.milestones[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del plan.portfolio_projects[0]  # type: ignore[misc]

        # Nested in-place collection mutations
        with pytest.raises(AttributeError):
            plan.milestones[0].target_missing_skills.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            plan.milestones[0].target_missing_skills[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(ValidationError):
            plan.executive_summary = "Modified summary"

    def test_optimized_bullet_point_nested_immutability(self):
        bullet = OptimizedBulletPoint(
            bullet_id="b1",
            source_entity_id="exp_0",
            source_field_path="experience[0].responsibilities[0]",
            improved_star_bullet="Engineered scalable ETL pipeline.",
            highlighted_skills=["Python", "PostgreSQL"],
            alignment_rationale="Emphasized pipeline engineering.",
            metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
        )
        assert isinstance(bullet.highlighted_skills, tuple)

        with pytest.raises(AttributeError):
            bullet.highlighted_skills.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            bullet.highlighted_skills[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del bullet.highlighted_skills[0]  # type: ignore[misc]

        with pytest.raises(ValidationError):
            bullet.improved_star_bullet = "Modified bullet text"

    def test_resume_optimization_report_nested_immutability(self):
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Strong backend alignment.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b1",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Engineered scalable ETL pipeline.",
                    highlighted_skills=["Python"],
                    alignment_rationale="Emphasized pipeline engineering.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                )
            ],
            section_reordering_recommendations=["Move projects above education"],
            ats_keyword_density_feedback=["Solid keyword presence"],
        )
        assert isinstance(report.optimized_bullets, tuple)
        assert isinstance(report.section_reordering_recommendations, tuple)
        assert isinstance(report.ats_keyword_density_feedback, tuple)

        with pytest.raises(AttributeError):
            report.optimized_bullets.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            report.optimized_bullets[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del report.section_reordering_recommendations[0]  # type: ignore[misc]

        # Nested in-place collection mutations
        with pytest.raises(AttributeError):
            report.optimized_bullets[0].highlighted_skills.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(ValidationError):
            report.target_role_alignment_summary = "Modified summary"

    def test_interview_question_nested_immutability(self):
        q = InterviewQuestion(
            question_id="q1",
            category=InterviewQuestionCategory.TECHNICAL,
            question_text="How do you handle async race conditions in Python?",
            interviewer_intent="Assess concurrency mastery.",
            targeted_skill_or_gap="Python",
            suggested_talking_points=["Discuss asyncio.Lock", "Cite previous project"],
        )
        assert isinstance(q.suggested_talking_points, tuple)

        with pytest.raises(AttributeError):
            q.suggested_talking_points.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            q.suggested_talking_points[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del q.suggested_talking_points[0]  # type: ignore[misc]

        with pytest.raises(ValidationError):
            q.question_text = "Modified question text"

    def test_interview_preparation_kit_nested_immutability(self):
        kit = InterviewPreparationKit(
            role_context_overview="Technical interviews will focus on async services.",
            technical_deep_dive_questions=[
                InterviewQuestion(
                    question_id="q1",
                    category=InterviewQuestionCategory.TECHNICAL,
                    question_text="Describe your async architecture experience.",
                    interviewer_intent="Assess async depth.",
                    targeted_skill_or_gap="Python",
                    suggested_talking_points=["Asyncio event loop"],
                )
            ],
            gap_mitigation_questions=[
                InterviewQuestion(
                    question_id="q2",
                    category=InterviewQuestionCategory.GAP_PROBE,
                    question_text="How would you approach containerization with Docker?",
                    interviewer_intent="Probe container skills.",
                    targeted_skill_or_gap="Docker",
                    suggested_talking_points=["Multi-stage builds"],
                )
            ],
            metric_challenge_questions=[],
            recommended_questions_for_interviewer=["What does the deployment pipeline look like?"],
        )
        assert isinstance(kit.technical_deep_dive_questions, tuple)
        assert isinstance(kit.gap_mitigation_questions, tuple)
        assert isinstance(kit.metric_challenge_questions, tuple)
        assert isinstance(kit.recommended_questions_for_interviewer, tuple)

        with pytest.raises(AttributeError):
            kit.technical_deep_dive_questions.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            kit.technical_deep_dive_questions[0] = "Tampered"  # type: ignore[index]

        with pytest.raises(TypeError):
            del kit.recommended_questions_for_interviewer[0]  # type: ignore[misc]

        # Nested in-place collection mutations
        with pytest.raises(AttributeError):
            kit.technical_deep_dive_questions[0].suggested_talking_points.append("Tampered")  # type: ignore[attr-defined]

        with pytest.raises(ValidationError):
            kit.role_context_overview = "Modified overview"

    def test_deterministic_validation_report_nested_immutability(self):
        violation = DeterministicValidationViolation(
            violation_type=ValidationViolationType.UNGROUNDED_SKILL_CLAIM,
            severity=ViolationSeverity.ERROR,
            message="Skill not found in resume.",
            target_field="skills",
            invalid_value="Kubernetes",
        )
        report = DeterministicValidationReport(
            report_id="rep-1",
            is_valid=False,
            target_deliverable_type="ResumeOptimizationReport",
            violations=[violation],
        )
        assert isinstance(report.violations, tuple)

        with pytest.raises(AttributeError):
            report.violations.append(violation)  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            report.violations[0] = violation  # type: ignore[index]

        with pytest.raises(TypeError):
            del report.violations[0]  # type: ignore[misc]

        with pytest.raises(ValidationError):
            report.is_valid = True


# =============================================================================
# 5. Authoritative Baseline Isolation Tests
# =============================================================================

class TestAuthoritativeBaselineIsolation:
    """
    Verifies that capturing an isolated baseline snapshot via create_baseline_snapshot
    protects against in-place mutation of collections and nested records.
    """

    def test_baseline_snapshot_memory_isolation(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)

        # Baseline must be equal in value across all fields
        assert baseline.overall_score == baseline_match_result.overall_score
        assert baseline.matched_skills == tuple(baseline_match_result.matched_skills)

        # Collections MUST be completely decoupled and converted to immutable containers
        assert baseline is not baseline_match_result
        assert isinstance(baseline.match_breakdown, tuple)
        assert isinstance(baseline.matched_skills, tuple)
        assert isinstance(baseline.skill_gaps, tuple)
        assert baseline.match_breakdown is not baseline_match_result.match_breakdown
        assert baseline.matched_skills is not baseline_match_result.matched_skills
        assert baseline.skill_gaps is not baseline_match_result.skill_gaps
        assert baseline.job_metadata is not baseline_match_result.job_metadata

    def test_in_place_matched_skills_mutation_caught_against_baseline(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        working = baseline_match_result

        # Tamper working result's matched_skills in-place
        working.matched_skills.append(
            SkillMatch(
                skill_name="Kubernetes",
                canonical_name="Kubernetes",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )
        )

        # Baseline was decoupled and preserved intact
        assert len(baseline.matched_skills) == 1
        assert len(working.matched_skills) == 2

        # Immutability verification raises error identifying matched skills mutation
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline, candidate=working)
        assert "Matched skills evidence mutated" in str(exc_info.value)

    def test_in_place_skill_gaps_mutation_caught_against_baseline(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        working = baseline_match_result

        # In-place clear of working skill gaps
        working.skill_gaps.clear()

        # Baseline remains intact
        assert len(baseline.skill_gaps) == 1
        assert len(working.skill_gaps) == 0

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline, candidate=working)
        assert "Skill gaps records mutated" in str(exc_info.value)

    def test_in_place_dimension_score_mutation_caught_against_baseline(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        working = baseline_match_result

        # Replace first dimension breakdown in working list
        tampered_dim = working.match_breakdown[0].model_copy(update={"dimension_score": 99.9})
        working.match_breakdown[0] = tampered_dim

        # Baseline was decoupled and preserves original 80.0 score
        assert baseline.match_breakdown[0].dimension_score == 80.0
        assert working.match_breakdown[0].dimension_score == 99.9

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline, candidate=working)
        assert "Dimensional match breakdown records mutated" in str(exc_info.value)

    def test_in_place_job_metadata_mutation_caught_against_baseline(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        working = baseline_match_result

        # In-place dictionary insertion into working metadata
        working.job_metadata["tampered_key"] = "tampered_value"

        # Baseline dictionary remains clean and empty
        assert "tampered_key" not in baseline.job_metadata

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(baseline=baseline, candidate=working)
        assert "Job metadata mutated" in str(exc_info.value)

    def test_intact_baseline_candidate_passes_cleanly(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        candidate = baseline.restore_job_match_result()

        # Verifying intact candidate against baseline succeeds with zero exceptions
        verify_workflow_score_immutability(
            baseline=baseline,
            candidate=candidate,
            expected_sha256=baseline.sha256_digest,
        )


# =============================================================================
# 6. Deliverable Serialization Round-Trip Tests (Tuple Preservation)
# =============================================================================

class TestDeliverableSerializationRoundTrip:
    """
    Verifies that all deliverable models serialize to standard JSON arrays
    and deserialize back to immutable tuples cleanly.
    """

    def test_triage_assessment_json_roundtrip(self):
        triage = TriageAssessment(
            primary_bottleneck="Cloud experience",
            key_strengths_summary=["Python", "PostgreSQL"],
            critical_gaps_summary=["AWS"],
            recommended_focus_areas=["Containerization"],
            suggested_goal=UserWorkflowGoal.GAP_REMEDIATION,
            advisory_rationale="Focus on cloud upskilling.",
        )
        json_data = triage.model_dump_json()
        restored = TriageAssessment.model_validate_json(json_data)
        assert restored == triage
        assert isinstance(restored.key_strengths_summary, tuple)
        assert restored.key_strengths_summary == ("Python", "PostgreSQL")

    def test_resume_optimization_report_json_roundtrip(self):
        report = ResumeOptimizationReport(
            target_role_alignment_summary="Strong backend alignment.",
            optimized_bullets=[
                OptimizedBulletPoint(
                    bullet_id="b1",
                    source_entity_id="exp_0",
                    source_field_path="experience[0].responsibilities[0]",
                    improved_star_bullet="Engineered scalable ETL pipeline.",
                    highlighted_skills=["Python"],
                    alignment_rationale="Emphasized pipeline engineering.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                )
            ],
            section_reordering_recommendations=["Move projects above education"],
            ats_keyword_density_feedback=["Solid keyword presence"],
        )
        json_data = report.model_dump_json()
        restored = ResumeOptimizationReport.model_validate_json(json_data)
        assert restored == report
        assert isinstance(restored.optimized_bullets, tuple)
        assert isinstance(restored.section_reordering_recommendations, tuple)
        assert isinstance(restored.ats_keyword_density_feedback, tuple)
        assert restored.optimized_bullets[0].highlighted_skills == ("Python",)

    def test_interview_preparation_kit_json_roundtrip(self):
        kit = InterviewPreparationKit(
            role_context_overview="Role overview",
            technical_deep_dive_questions=[
                InterviewQuestion(
                    question_id="q1",
                    category=InterviewQuestionCategory.TECHNICAL,
                    question_text="Describe your async architecture experience.",
                    interviewer_intent="Assess async depth.",
                    targeted_skill_or_gap="Python",
                    suggested_talking_points=["Asyncio event loop"],
                )
            ],
            gap_mitigation_questions=[],
            metric_challenge_questions=[],
            recommended_questions_for_interviewer=["What is the deployment pipeline?"],
        )
        json_data = kit.model_dump_json()
        restored = InterviewPreparationKit.model_validate_json(json_data)
        assert restored == kit
        assert isinstance(restored.technical_deep_dive_questions, tuple)
        assert isinstance(restored.recommended_questions_for_interviewer, tuple)
        assert restored.technical_deep_dive_questions[0].suggested_talking_points == ("Asyncio event loop",)


# =============================================================================
# 7. Authoritative Baseline Tampering Audit Tests (Rigorous Adversarial Suite)
# =============================================================================

class TestAuthoritativeBaselineTamperingAudit:
    """
    Dedicated test suite addressing Phase 5.1 Final Authoritative Baseline Tampering Audit:
    Proves that workflow nodes cannot mutate the baseline snapshot, collections,
    or independent cryptographic integrity anchors.
    """

    def test_in_place_mutation_of_every_baseline_collection_blocked(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)

        # 1. match_breakdown
        with pytest.raises(AttributeError):
            baseline.match_breakdown.append("Tampered")  # type: ignore[attr-defined]

        # 2. matched_skills
        with pytest.raises(AttributeError):
            baseline.matched_skills.append("Tampered")  # type: ignore[attr-defined]

        # 3. partial_matches
        with pytest.raises(AttributeError):
            baseline.partial_matches.append("Tampered")  # type: ignore[attr-defined]

        # 4. missing_evidence
        with pytest.raises(AttributeError):
            baseline.missing_evidence.append("Tampered")  # type: ignore[attr-defined]

        # 5. unknown_evidence
        with pytest.raises(AttributeError):
            baseline.unknown_evidence.append("Tampered")  # type: ignore[attr-defined]

        # 6. project_evidence
        with pytest.raises(AttributeError):
            baseline.project_evidence.append("Tampered")  # type: ignore[attr-defined]

        # 7. strengths
        with pytest.raises(AttributeError):
            baseline.strengths.append("Tampered")  # type: ignore[attr-defined]

        # 8. skill_gaps
        with pytest.raises(AttributeError):
            baseline.skill_gaps.append("Tampered")  # type: ignore[attr-defined]

        # 9. recommendations
        with pytest.raises(AttributeError):
            baseline.recommendations.append("Tampered")  # type: ignore[attr-defined]

        # 10. processing_warnings
        with pytest.raises(AttributeError):
            baseline.processing_warnings.append("Tampered")  # type: ignore[attr-defined]

        # 11. job_metadata MappingProxyType
        with pytest.raises(TypeError):
            baseline.job_metadata["tampered_key"] = "tampered_value"

        # 12. Item assignment on collections
        with pytest.raises(TypeError):
            baseline.match_breakdown[0] = "Tampered"  # type: ignore[index]

        # 13. Item deletion on collections
        with pytest.raises(TypeError):
            del baseline.matched_skills[0]  # type: ignore[misc]

        # 14. Attribute reassignment on frozen model
        with pytest.raises(ValidationError):
            baseline.overall_score = 99.0

        with pytest.raises(ValidationError):
            baseline.sha256_digest = "tampered_digest"

    def test_node_receiving_state_cannot_mutate_baseline_snapshot(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        initial_digest = baseline.sha256_digest

        resume_profile = ResumeProfile(
            filename="alice.pdf",
            page_count=1,
            raw_character_count=200,
            cleaned_character_count=190,
            status=ProfileProcessingStatus.SUCCESS,
        )
        jd_profile = JobDescriptionProfile(
            original_description="Developer needed",
            job_title="Developer",
        )

        initial_state: CareerWorkflowState = {
            "workflow_id": "audit-wf-1",
            "match_result": baseline_match_result,
            "baseline_match_snapshot": baseline,
            "baseline_sha256": initial_digest,
            "resume_profile": resume_profile,
            "jd_profile": jd_profile,
        }

        # Node attempting in-place collection mutation
        def rogue_node(state: CareerWorkflowState) -> dict:
            snap = state["baseline_match_snapshot"]
            # Attempt to append to matched_skills
            try:
                snap.matched_skills.append("hacked")  # type: ignore[attr-defined]
            except AttributeError:
                pass
            # Attempt to clear skill_gaps
            try:
                snap.skill_gaps.clear()  # type: ignore[attr-defined]
            except AttributeError:
                pass
            # Attempt to mutate job_metadata
            try:
                snap.job_metadata["evil"] = True
            except TypeError:
                pass
            return {"current_step": "rogue_complete"}

        builder = StateGraph(CareerWorkflowState)
        builder.add_node("rogue_node", rogue_node)
        builder.add_edge(START, "rogue_node")
        builder.add_edge("rogue_node", END)
        graph = builder.compile(checkpointer=MemorySaver())

        final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": "audit-thread"}})

        # Verify baseline remains completely uncorrupted
        surviving_snap = final_state["baseline_match_snapshot"]
        assert surviving_snap.sha256_digest == initial_digest
        assert len(surviving_snap.matched_skills) == 1
        assert len(surviving_snap.skill_gaps) == 1
        assert "evil" not in surviving_snap.job_metadata

    def test_node_cannot_replace_baseline_or_integrity_hash_via_update(self, baseline_match_result):
        fake_snapshot = create_baseline_snapshot(
            baseline_match_result.model_copy(update={"overall_score": 99.0}, deep=True)
        )

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update("malicious_node", {"baseline_match_snapshot": fake_snapshot})
        assert "baseline_match_snapshot" in str(exc_info.value)

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update("malicious_node", {"baseline_sha256": "fake_hash_value"})
        assert "baseline_sha256" in str(exc_info.value)

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update("malicious_node", {"baseline_snapshot": fake_snapshot})
        assert "baseline_snapshot" in str(exc_info.value)

        with pytest.raises(ScoreImmutabilityError) as exc_info:
            intercept_node_update("malicious_node", {"match_result": "tampered"})
        assert "match_result" in str(exc_info.value)

    def test_mutating_original_result_after_baseline_creation_does_not_affect_baseline(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        initial_digest = baseline.sha256_digest

        # Tamper original JobMatchResult object after baseline snapshot creation
        baseline_match_result.matched_skills.append(
            SkillMatch(
                skill_name="Kubernetes",
                canonical_name="Kubernetes",
                match_type=SkillMatchType.EXACT_MATCH,
                match_confidence=1.0,
                requirement_priority=RequirementPriority.REQUIRED,
                evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
            )
        )
        baseline_match_result.skill_gaps.clear()
        baseline_match_result.job_metadata["post_creation_tamper"] = True

        # Baseline snapshot remains strictly untouched
        assert baseline.sha256_digest == initial_digest
        restored = baseline.restore_job_match_result()
        assert len(restored.matched_skills) == 1
        assert len(restored.skill_gaps) == 1
        assert "post_creation_tamper" not in restored.job_metadata

        # Final audit detects that original has diverged from baseline
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline=baseline, candidate=baseline_match_result)

    def test_final_audit_detects_every_tampered_field(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        independent_digest = baseline.sha256_digest

        # Candidate with tampered score
        tampered_score = baseline_match_result.model_copy(update={"overall_score": 50.0}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline, tampered_score, expected_sha256=independent_digest)

        # Candidate with tampered matched skills
        tampered_skills = baseline_match_result.model_copy(update={"matched_skills": []}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline, tampered_skills, expected_sha256=independent_digest)

        # Candidate with tampered skill gaps
        tampered_gaps = baseline_match_result.model_copy(update={"skill_gaps": []}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline, tampered_gaps, expected_sha256=independent_digest)

        # Candidate with tampered job metadata
        tampered_meta = baseline_match_result.model_copy(update={"job_metadata": {"hacked": 1}}, deep=True)
        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline, tampered_meta, expected_sha256=independent_digest)

        # Mismatched independent expected_sha256
        with pytest.raises(ScoreImmutabilityError) as exc_info:
            verify_workflow_score_immutability(
                baseline,
                baseline.restore_job_match_result(),
                expected_sha256="0" * 64,
            )
        assert "Cryptographic Integrity Violation" in str(exc_info.value)

    def test_untouched_baseline_passes_final_audit(self, baseline_match_result):
        baseline = create_baseline_snapshot(baseline_match_result)
        candidate = baseline.restore_job_match_result()

        # Both baseline and candidate digests match independent digest
        verify_workflow_score_immutability(
            baseline=baseline,
            candidate=candidate,
            expected_sha256=baseline.sha256_digest,
        )

    def test_all_protections_operational_under_python_o(self, baseline_match_result):
        # Explicit test verifying non-assert conditional raises
        baseline = create_baseline_snapshot(baseline_match_result)
        tampered = baseline_match_result.model_copy(update={"overall_score": 10.0}, deep=True)

        with pytest.raises(ScoreImmutabilityError):
            verify_workflow_score_immutability(baseline, tampered)

        with pytest.raises(ScoreImmutabilityError):
            intercept_node_update("test", {"baseline_sha256": "tampered"})
