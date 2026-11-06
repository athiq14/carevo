"""
Tests for Streamlit Agentic Workflow UI integration.
Verifies UI state initialization, HITL checkpoints, deliverable rendering,
final approval actions, and workflow reset functionality.
"""
from unittest.mock import MagicMock, patch
import pytest

from core.workflow.coordinator import WorkflowExecutionCoordinator
from core.workflow.ui import get_or_create_coordinator, render_agentic_workflow_ui
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
from models.workflow import UserWorkflowGoal


@pytest.fixture
def sample_match_and_profile():
    profile = ResumeProfile(
        filename="alice_resume.pdf",
        page_count=1,
        raw_character_count=800,
        cleaned_character_count=780,
        status=ProfileProcessingStatus.SUCCESS,
        skills=[
            ExtractedSkill(name="Python", original_mention="Python", category=SkillCategory.PROGRAMMING_LANGUAGES),
            ExtractedSkill(name="Docker", original_mention="Docker", category=SkillCategory.DEVOPS_AND_INFRASTRUCTURE),
        ],
        experience=[
            WorkExperience(
                job_title="Software Engineer",
                company="Tech Corp",
                employment_start="2021",
                employment_end="2024",
                responsibilities=["Developed backend services with Python."],
                technologies=["Python", "Docker"],
                original_text="Software Engineer @ Tech Corp",
            )
        ],
        education=[],
        projects=[],
        has_ambiguities=False,
    )
    match = JobMatchResult(
        overall_score=75.0,
        fit_classification=CandidateFitLevel.STRONG,
        evaluation_status=EvaluationStatus.FULL_EVALUATION,
        active_evaluation_coverage=85.0,
        unassessed_reserve=15.0,
        match_breakdown=[
            MatchBreakdown(
                dimension_name="Technical Skills",
                dimension_score=75.0,
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
        partial_matches=[],
        missing_evidence=[
            SkillGap(
                skill_name="Kubernetes",
                priority=RequirementPriority.PREFERRED,
                gap_explanation="Missing Kubernetes",
                suggested_learning_direction="Helm charts",
            )
        ],
        unknown_evidence=[],
        project_evidence=[],
        strengths=["Strong Python"],
        skill_gaps=[],
        recommendations=[],
        processing_warnings=[],
        job_metadata={"job_title": "Backend Engineer", "company": "Tech Corp"},
    )
    return match, profile


def test_get_or_create_coordinator(monkeypatch):
    """Verify coordinator is lazily initialized and stored in session_state."""
    fake_session = {}
    monkeypatch.setattr("streamlit.session_state", fake_session)

    c1 = get_or_create_coordinator()
    assert isinstance(c1, WorkflowExecutionCoordinator)
    assert fake_session["workflow_coordinator"] is c1

    c2 = get_or_create_coordinator()
    assert c2 is c1


def test_render_ui_initial_state_prompts_to_launch(sample_match_and_profile, monkeypatch):
    """Verify initial state renders launch button and radio options."""
    match, profile = sample_match_and_profile
    fake_session = {}
    monkeypatch.setattr("streamlit.session_state", fake_session)

    # Mock streamlit components
    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    monkeypatch.setattr("streamlit.info", MagicMock())
    mock_radio = MagicMock(return_value="FULL_PREPARATION")
    monkeypatch.setattr("streamlit.radio", mock_radio)
    mock_button = MagicMock(return_value=False)
    monkeypatch.setattr("streamlit.button", mock_button)
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    render_agentic_workflow_ui(match_result=match, resume_profile=profile)

    mock_radio.assert_called_once()
    assert "active_workflow_thread_id" not in fake_session


def test_render_ui_handles_clarification_checkpoint(sample_match_and_profile, monkeypatch):
    """Verify UI presents clarification prompt when workflow is interrupted for clarification."""
    match, profile = sample_match_and_profile
    fake_session = {
        "active_workflow_thread_id": "thread_test_clar",
        "workflow_state": {
            "current_step": "clarification",
            "completed_nodes": ["triage"],
        },
        "workflow_interrupts": [
            {
                "type": "clarification_required",
                "stage": "clarification",
                "message": "High unassessed reserve. Please clarify.",
            }
        ],
    }
    monkeypatch.setattr("streamlit.session_state", fake_session)

    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    mock_warning = MagicMock()
    monkeypatch.setattr("streamlit.warning", mock_warning)
    monkeypatch.setattr("streamlit.markdown", MagicMock())
    monkeypatch.setattr("streamlit.text_area", MagicMock(return_value="Backend specialist"))
    mock_button = MagicMock(return_value=False)
    monkeypatch.setattr("streamlit.button", mock_button)
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    render_agentic_workflow_ui(match_result=match, resume_profile=profile)

    # Verified warning was displayed for clarification checkpoint
    assert any("Candidate Clarification Required" in str(c) for c in mock_warning.call_args_list)


def test_render_ui_handles_final_approval_checkpoint(sample_match_and_profile, monkeypatch):
    """Verify UI displays approval/rejection buttons during final approval interrupt."""
    match, profile = sample_match_and_profile
    fake_session = {
        "active_workflow_thread_id": "thread_test_appr",
        "workflow_state": {
            "current_step": "final_approval",
            "completed_nodes": ["triage", "goal_selection", "remediation", "deterministic_audit"],
            "audit_passed": True,
            "remediation_plan": None,
            "resume_optimization": None,
            "interview_kit": None,
        },
        "workflow_interrupts": [
            {
                "type": "final_approval_required",
                "stage": "final_approval",
                "audit_passed": True,
            }
        ],
    }
    monkeypatch.setattr("streamlit.session_state", fake_session)

    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    mock_success = MagicMock()
    monkeypatch.setattr("streamlit.success", mock_success)
    monkeypatch.setattr("streamlit.markdown", MagicMock())
    monkeypatch.setattr("streamlit.button", MagicMock(return_value=False))
    monkeypatch.setattr("streamlit.tabs", MagicMock(return_value=[MagicMock()] * 5))
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    render_agentic_workflow_ui(match_result=match, resume_profile=profile)

    assert any("Deliverables Ready for Final Review" in str(c) for c in mock_success.call_args_list)


# ---------------------------------------------------------------------------
# Phase 5.5 Regression Tests: Canonical Class Identity & Type Safety
# ---------------------------------------------------------------------------

def test_pipeline_output_accepted_by_workflow_ui(sample_match_and_profile, monkeypatch):
    """Verify that match result from pipeline is accepted by workflow UI entrypoint without TypeError."""
    match, profile = sample_match_and_profile
    fake_session = {}
    monkeypatch.setattr("streamlit.session_state", fake_session)
    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    monkeypatch.setattr("streamlit.radio", MagicMock(return_value=None))
    monkeypatch.setattr("streamlit.button", MagicMock(return_value=False))
    monkeypatch.setattr("streamlit.info", MagicMock())
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    # Should execute cleanly without raising TypeError
    render_agentic_workflow_ui(match_result=match, resume_profile=profile)


def test_reloaded_job_match_result_identity_normalization(sample_match_and_profile):
    """
    Simulate module reload where JobMatchResult class identity in memory changes.
    Verify ensure_canonical_job_match_result, create_baseline_snapshot, and
    serialize_authoritative_match_result normalize it to the canonical class without TypeError.
    """
    import importlib
    import models.matching
    from core.workflow.validators import (
        create_baseline_snapshot,
        ensure_canonical_job_match_result,
        serialize_authoritative_match_result,
    )
    from models.workflow import AuthoritativeMatchSnapshot

    match, _ = sample_match_and_profile
    # Capture original dump
    original_score = match.overall_score

    # Recreate an instance using a dynamically constructed class with matching name
    # to perfectly simulate a reloaded class identity with distinct id(cls)
    importlib.reload(models.matching)
    from models.matching import JobMatchResult as ReloadedJobMatchResult

    reloaded_instance = ReloadedJobMatchResult.model_validate(match.model_dump())

    # 1. ensure_canonical_job_match_result returns valid canonical instance
    canonical_match = ensure_canonical_job_match_result(reloaded_instance)
    assert canonical_match.overall_score == original_score
    assert isinstance(canonical_match, ReloadedJobMatchResult)

    # 2. create_baseline_snapshot succeeds and produces AuthoritativeMatchSnapshot
    snapshot = create_baseline_snapshot(reloaded_instance)
    assert isinstance(snapshot, AuthoritativeMatchSnapshot)
    assert snapshot.overall_score == original_score
    assert len(snapshot.sha256_digest) == 64

    # 3. serialize_authoritative_match_result succeeds without TypeError
    canonical_json = serialize_authoritative_match_result(reloaded_instance)
    assert isinstance(canonical_json, str)
    assert len(canonical_json) > 0


def test_invalid_types_strictly_rejected_with_type_error(sample_match_and_profile):
    """
    Verify completely invalid types and corrupted schema objects are strictly rejected
    with TypeError across all validation entrypoints.
    """
    from pydantic import BaseModel
    from core.workflow.validators import (
        create_baseline_snapshot,
        ensure_canonical_job_match_result,
        serialize_authoritative_match_result,
        verify_workflow_score_immutability,
    )

    match, profile = sample_match_and_profile

    # Arbitrary primitive and collection types
    invalid_inputs = [{}, [], "random_string", 12345, 99.9, None]

    for inv in invalid_inputs:
        with pytest.raises(TypeError):
            ensure_canonical_job_match_result(inv)

        with pytest.raises(TypeError):
            create_baseline_snapshot(inv)

        with pytest.raises(TypeError):
            serialize_authoritative_match_result(inv)

        with pytest.raises(TypeError):
            verify_workflow_score_immutability(baseline=match, candidate=inv)

        with pytest.raises(TypeError):
            render_agentic_workflow_ui(match_result=inv, resume_profile=profile)

    # Arbitrary object claiming class name JobMatchResult but lacking valid schema
    class SpoofedJobMatchResult(BaseModel):
        injected: str = "malicious"

    spoofed = SpoofedJobMatchResult()
    spoofed.__class__.__name__ = "JobMatchResult"

    with pytest.raises(TypeError):
        ensure_canonical_job_match_result(spoofed)

    with pytest.raises(TypeError):
        create_baseline_snapshot(spoofed)


# ---------------------------------------------------------------------------
# Phase 5.6 Regression Tests: DuplicateElementId Prevention & Action Verification
# ---------------------------------------------------------------------------

def test_multiple_ui_renderings_no_duplicate_element_ids(sample_match_and_profile, monkeypatch):
    """
    Simulate rendering render_agentic_workflow_ui twice in a single render pass
    (Tab 2 with key_prefix='tab2_agentic' and Tab 4 with key_prefix='tab4_agentic').
    Verify that every widget (button, radio, text_area, download_button) receives an
    explicit, unique, stable key with zero collisions.
    """
    match, profile = sample_match_and_profile
    fake_session = {}
    monkeypatch.setattr("streamlit.session_state", fake_session)
    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    monkeypatch.setattr("streamlit.info", MagicMock())
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    registered_keys = []

    def mock_radio(*args, **kwargs):
        key = kwargs.get("key")
        assert key is not None, f"st.radio called without explicit key: {args}"
        assert key not in registered_keys, f"Duplicate element ID detected for st.radio: {key}"
        registered_keys.append(key)
        return "FULL_PREPARATION"

    def mock_button(*args, **kwargs):
        key = kwargs.get("key")
        assert key is not None, f"st.button called without explicit key: {args}"
        assert key not in registered_keys, f"Duplicate element ID detected for st.button: {key}"
        registered_keys.append(key)
        return False

    monkeypatch.setattr("streamlit.radio", mock_radio)
    monkeypatch.setattr("streamlit.button", mock_button)

    # Render Tab 2 instance
    render_agentic_workflow_ui(match_result=match, resume_profile=profile, key_prefix="tab2_agentic")

    # Render Tab 4 instance in same pass
    render_agentic_workflow_ui(match_result=match, resume_profile=profile, key_prefix="tab4_agentic")

    # Verify both instances registered unique keys
    assert "tab2_agentic_reset_workflow_btn" in registered_keys
    assert "tab4_agentic_reset_workflow_btn" in registered_keys
    assert "tab2_agentic_initial_goal_radio" in registered_keys
    assert "tab4_agentic_initial_goal_radio" in registered_keys
    assert "tab2_agentic_launch_workflow_btn" in registered_keys
    assert "tab4_agentic_launch_workflow_btn" in registered_keys
    assert len(registered_keys) == len(set(registered_keys))


def test_workflow_reset_action_preserves_authoritative_match_result(sample_match_and_profile, monkeypatch):
    """
    Verify that Reset Workflow clears active workflow state, notifies coordinator,
    calls st.rerun(), and preserves the Phase 4 match result and SHA-256 digest completely intact.
    """
    from core.workflow.validators import compute_match_result_digest
    match, profile = sample_match_and_profile
    original_score = match.overall_score
    original_digest = compute_match_result_digest(match)

    mock_coordinator = MagicMock()
    fake_session = {
        "workflow_coordinator": mock_coordinator,
        "active_workflow_thread_id": "thread_reset_test",
        "workflow_state": {"current_step": "triage", "match_result": match},
        "workflow_interrupts": [{"type": "clarification_required"}],
        "pipeline_result": MagicMock(match_result=match),
    }
    monkeypatch.setattr("streamlit.session_state", fake_session)
    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    # Simulate clicking Reset button
    def mock_button(*args, **kwargs):
        if kwargs.get("key") == "tab2_agentic_reset_workflow_btn":
            return True
        return False

    mock_rerun = MagicMock()
    monkeypatch.setattr("streamlit.button", mock_button)
    monkeypatch.setattr("streamlit.rerun", mock_rerun)

    render_agentic_workflow_ui(match_result=match, resume_profile=profile, key_prefix="tab2_agentic")

    # 1. Coordinator reset called with thread ID
    mock_coordinator.reset_workflow.assert_called_once_with("thread_reset_test")

    # 2. Workflow transient state purged
    assert "active_workflow_thread_id" not in fake_session
    assert "workflow_state" not in fake_session
    assert "workflow_interrupts" not in fake_session

    # 3. Streamlit rerun requested
    mock_rerun.assert_called_once()

    # 4. Phase 4 match result completely untampered
    assert "pipeline_result" in fake_session
    assert fake_session["pipeline_result"].match_result.overall_score == original_score
    assert compute_match_result_digest(fake_session["pipeline_result"].match_result) == original_digest


def test_final_approval_actions_and_unique_keys(sample_match_and_profile, monkeypatch):
    """
    Verify that Approve, Request Changes, and Reject buttons all have unique keys
    and correctly deliver the expected resume value to coordinator.resume_workflow.
    """
    match, profile = sample_match_and_profile
    mock_coordinator = MagicMock()
    mock_coordinator.resume_workflow.return_value = ({"current_step": "completed"}, [])

    for decision_btn_key, expected_val in [
        ("tab2_agentic_approve_deliverables_btn", "APPROVE"),
        ("tab2_agentic_request_changes_btn", "REQUEST_CHANGES"),
        ("tab2_agentic_reject_deliverables_btn", "REJECT"),
    ]:
        fake_session = {
            "workflow_coordinator": mock_coordinator,
            "active_workflow_thread_id": "thread_appr_actions",
            "workflow_state": {
                "current_step": "final_approval",
                "completed_nodes": ["triage", "goal_selection", "remediation", "deterministic_audit"],
                "audit_passed": True,
            },
            "workflow_interrupts": [
                {
                    "type": "final_approval_required",
                    "stage": "final_approval",
                    "audit_passed": True,
                }
            ],
        }
        monkeypatch.setattr("streamlit.session_state", fake_session)
        monkeypatch.setattr("streamlit.divider", MagicMock())
        monkeypatch.setattr("streamlit.header", MagicMock())
        monkeypatch.setattr("streamlit.caption", MagicMock())
        monkeypatch.setattr("streamlit.success", MagicMock())
        monkeypatch.setattr("streamlit.markdown", MagicMock())
        monkeypatch.setattr("streamlit.tabs", MagicMock(return_value=[MagicMock()] * 5))
        mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
        monkeypatch.setattr("streamlit.columns", mock_columns)

        def mock_button(*args, **kwargs):
            return kwargs.get("key") == decision_btn_key

        mock_rerun = MagicMock()
        monkeypatch.setattr("streamlit.button", mock_button)
        monkeypatch.setattr("streamlit.rerun", mock_rerun)

        render_agentic_workflow_ui(match_result=match, resume_profile=profile, key_prefix="tab2_agentic")

        mock_coordinator.resume_workflow.assert_called_with(
            thread_id="thread_appr_actions",
            resume_value=expected_val,
        )
        mock_rerun.assert_called_once()


def test_completed_workflow_results_visibility_and_download(sample_match_and_profile, monkeypatch):
    """
    Verify completed workflow renders celebration banner, final approval status,
    deliverables tabs, and download button with unique key.
    """
    match, profile = sample_match_and_profile
    fake_session = {
        "active_workflow_thread_id": "thread_completed_test",
        "workflow_state": {
            "current_step": "completed",
            "completed_nodes": ["triage", "goal_selection", "remediation", "synthesis"],
            "final_approval_decision": "APPROVE",
            "audit_passed": True,
            "consolidated_report": {
                "report_id": "test_report_123",
                "overall_score": 75.0,
            },
        },
        "workflow_interrupts": [],
    }
    monkeypatch.setattr("streamlit.session_state", fake_session)
    monkeypatch.setattr("streamlit.divider", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr("streamlit.caption", MagicMock())
    mock_success = MagicMock()
    monkeypatch.setattr("streamlit.success", mock_success)
    mock_write = MagicMock()
    monkeypatch.setattr("streamlit.write", mock_write)
    monkeypatch.setattr("streamlit.subheader", MagicMock())
    monkeypatch.setattr("streamlit.markdown", MagicMock())
    monkeypatch.setattr("streamlit.button", MagicMock(return_value=False))
    monkeypatch.setattr("streamlit.tabs", MagicMock(return_value=[MagicMock()] * 5))
    mock_columns = lambda spec: [MagicMock() for _ in range(len(spec) if isinstance(spec, list) else int(spec))]
    monkeypatch.setattr("streamlit.columns", mock_columns)

    download_called = []
    def mock_download_button(*args, **kwargs):
        download_called.append(kwargs)
        return False

    monkeypatch.setattr("streamlit.download_button", mock_download_button)

    render_agentic_workflow_ui(match_result=match, resume_profile=profile, key_prefix="tab2_agentic")

    # Assert completed message displayed
    assert any("Agentic Workflow Execution Completed" in str(c) for c in mock_success.call_args_list)
    assert any("APPROVE" in str(c) for c in mock_write.call_args_list)

    # Assert download button rendered with unique key
    assert len(download_called) == 1
    assert download_called[0]["key"] == "tab2_agentic_download_consolidated_report_btn"


def test_synthetic_demo_pdf_generation():
    """Verify in-memory synthetic demo candidate PDF generates valid PDF bytes."""
    from app import generate_synthetic_demo_pdf
    pdf_bytes = generate_synthetic_demo_pdf()
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")


def test_load_demo_showcase_populates_state(monkeypatch):
    """Verify 1-click demo showcase loads synthetic profile and target job description."""
    import streamlit as st
    from app import load_demo_showcase

    fake_session = {}
    monkeypatch.setattr(st, "session_state", fake_session)

    load_demo_showcase()

    assert fake_session.get("demo_showcase_active") is True
    assert fake_session.get("tab1_resume_filename") == "alex_rivera_demo_resume.pdf"
    assert "tab1_resume_profile" in fake_session
    profile = fake_session["tab1_resume_profile"]
    assert profile.filename == "alex_rivera_demo_resume.pdf"
    assert any("Python" in s.name for s in profile.skills)
    assert fake_session.get("input_jt_field") == "Senior Platform Engineer"
    assert fake_session.get("input_comp_field") == "Stripe"
    assert "Stripe" in fake_session.get("input_jd_area", "")


def test_compatibility_score_disclaimer_in_app_source():
    """Verify app.py explicitly clarifies that match score is not a hiring guarantee or probability."""
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()

    assert "This is an evidence alignment evaluation, not a hiring guarantee or probability." in code



