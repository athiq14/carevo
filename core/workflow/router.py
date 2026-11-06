"""
Deterministic routing logic for LangGraph workflow execution.
Implements exact routing tables for all permitted workflow goals,
conditional edge evaluation, and bounded correction branching.
"""
import re
from typing import Literal
from core.workflow.state import CareerWorkflowState
from models.workflow import UserWorkflowGoal


def route_after_triage(state: CareerWorkflowState) -> Literal["clarification", "goal_selection"]:
    """
    Evaluate whether candidate evidence ambiguity, unassessed reserve, or
    divergence between explicit career objective and target JD requires clarification.
    """
    match_result = state.get("match_result")
    profile = state.get("resume_profile")
    user_clarifications = state.get("user_clarifications") or {}
    user_career_obj = state.get("user_career_objective")
    jd_profile = state.get("jd_profile")

    # If already clarified, proceed directly to goal selection
    if user_clarifications:
        return "goal_selection"

    # Check for divergence/conflict between explicit user career objective and target JD
    if user_career_obj and jd_profile:
        target_role = (jd_profile.job_title or "").strip().lower()
        career_obj = user_career_obj.strip().lower()
        if target_role and career_obj and target_role != career_obj:
            target_words = set(re.findall(r'\b[a-z]{3,}\b', target_role))
            obj_words = set(re.findall(r'\b[a-z]{3,}\b', career_obj))
            stop_words = {"engineer", "developer", "senior", "lead", "associate", "specialist", "role", "position", "analyst", "advisor"}
            meaningful_target = target_words - stop_words
            meaningful_obj = obj_words - stop_words
            if meaningful_target and meaningful_obj and not (meaningful_target & meaningful_obj):
                # Substantive domain conflict (e.g. Insurance Advisor vs AI Builder / Platform Engineer)
                return "clarification"

    # Check unassessed reserve or active coverage thresholds
    if match_result is not None:
        unassessed_reserve = getattr(match_result, "unassessed_reserve", 0.0)
        active_coverage = getattr(match_result, "active_evaluation_coverage", 100.0)
        if unassessed_reserve > 30.0 or active_coverage < 70.0:
            return "clarification"

    # Check for ambiguous resume records
    if profile is not None and getattr(profile, "has_ambiguities", False):
        return "clarification"

    return "goal_selection"


def route_after_clarification(state: CareerWorkflowState) -> Literal["goal_selection"]:
    """Always transitions to goal_selection after clarification."""
    return "goal_selection"


def route_after_goal_selection(
    state: CareerWorkflowState,
) -> Literal["remediation", "resume_optimization", "interview_prep", "goal_selection"]:
    """
    Route execution according to the explicitly selected user_goal.
    Never allows an LLM to choose or override the user's goal.
    """
    user_goal = state.get("user_goal")

    if user_goal == UserWorkflowGoal.FULL_PREPARATION:
        return "remediation"
    elif user_goal == UserWorkflowGoal.GAP_REMEDIATION:
        return "remediation"
    elif user_goal == UserWorkflowGoal.RESUME_OPTIMIZATION:
        return "resume_optimization"
    elif user_goal == UserWorkflowGoal.INTERVIEW_DRILL:
        return "interview_prep"
    else:
        # Invalid or missing goal; re-prompt
        return "goal_selection"


def route_after_remediation(
    state: CareerWorkflowState,
) -> Literal["resume_optimization", "deterministic_audit"]:
    """
    Route after remediation:
    FULL_PREPARATION continues to resume_optimization.
    GAP_REMEDIATION proceeds directly to deterministic_audit.
    """
    if state.get("user_goal") == UserWorkflowGoal.FULL_PREPARATION:
        return "resume_optimization"
    return "deterministic_audit"


def route_after_resume_optimization(
    state: CareerWorkflowState,
) -> Literal["interview_prep", "deterministic_audit"]:
    """
    Route after resume_optimization:
    FULL_PREPARATION continues to interview_prep.
    RESUME_OPTIMIZATION proceeds directly to deterministic_audit.
    """
    if state.get("user_goal") == UserWorkflowGoal.FULL_PREPARATION:
        return "interview_prep"
    return "deterministic_audit"


def route_after_interview_prep(
    state: CareerWorkflowState,
) -> Literal["deterministic_audit"]:
    """Always transitions to deterministic_audit."""
    return "deterministic_audit"


def route_after_audit(
    state: CareerWorkflowState,
) -> Literal["final_approval", "correction"]:
    """
    Route after deterministic safety audit:
    - If audit passed: proceed to final human approval.
    - If audit failed and correction attempts < 1: proceed to bounded correction cycle.
    - If audit failed and correction exhausted (attempts >= 1): stop and request human review.
    """
    audit_passed = state.get("audit_passed", False)
    attempts = state.get("correction_attempt_count", 0)

    if audit_passed:
        return "final_approval"

    if attempts < 1:
        return "correction"

    # Bounded retry exhausted: route to human review via final_approval node
    return "final_approval"


def route_after_correction(
    state: CareerWorkflowState,
) -> Literal["deterministic_audit"]:
    """After correction attempt, re-audit the deliverables."""
    return "deterministic_audit"


def route_after_final_approval(
    state: CareerWorkflowState,
) -> Literal["synthesis", "__end__"]:
    """
    Route after final human approval checkpoint:
    - Explicit 'APPROVE' leads to synthesis and export.
    - 'REJECT' or 'REQUEST_CHANGES' terminates workflow without report synthesis.
    """
    decision = state.get("final_approval_decision")
    if decision == "APPROVE":
        return "synthesis"
    return "__end__"

