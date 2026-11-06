"""
LangGraph workflow nodes for Phase 5 Agentic Career Intelligence.
Implements modular node functions for Triage, Clarification (HITL),
Goal Selection (HITL), Remediation, Resume Optimization, Interview Preparation,
Deterministic Safety Audit, Bounded Correction, Final Approval (HITL), and Synthesis.
"""
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from langgraph.types import interrupt

from core.workflow.state import CareerWorkflowState
from core.workflow.validators import (
    DeterministicWorkflowValidator,
    verify_workflow_score_immutability,
)
from models.workflow import (
    DeterministicValidationReport,
    UserWorkflowGoal,
    WorkflowTraceEvent,
)

# Optional dependency injection hook for testing with MockLLMProvider
_AGENT_ENGINE_OVERRIDE: Optional[Any] = None


def set_agent_engine_override(engine: Optional[Any]) -> None:
    """Set global engine override for specialist agents during offline testing."""
    global _AGENT_ENGINE_OVERRIDE
    _AGENT_ENGINE_OVERRIDE = engine


def _get_agent_engine() -> Optional[Any]:
    """Retrieve active engine override or None for default production manager."""
    return _AGENT_ENGINE_OVERRIDE


def _make_trace_event(node_name: str, event_type: str, message: str) -> WorkflowTraceEvent:
    """Helper to construct immutable WorkflowTraceEvent."""
    return WorkflowTraceEvent(
        event_id=str(uuid.uuid4()),
        node_name=node_name,
        event_type=event_type,
        message=message,
        timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# Node 1: Triage Node
# =============================================================================

def triage_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Executes TriageAgent to diagnose match facts, identify bottlenecks,
    and suggest an advisory workflow goal.
    """
    if state.get("triage_assessment") is not None:
        # Preserve existing triage assessment if re-executed
        return {
            "current_step": "triage",
            "completed_nodes": ["triage"],
        }

    from agents.triage_agent import TriageAgent
    agent = TriageAgent(engine=_get_agent_engine())
    match_result = state["match_result"]
    jd_profile = state.get("jd_profile")

    assessment, telemetry = agent.run(match_result=match_result, jd_profile=jd_profile)

    tokens = telemetry.get("token_usage") or {}
    token_usage = {
        "prompt_tokens": tokens.get("prompt_tokens", 0) or 0,
        "completion_tokens": tokens.get("completion_tokens", 0) or 0,
        "total_tokens": tokens.get("total_tokens", 0) or 0,
    }

    event = _make_trace_event(
        node_name="triage",
        event_type="NODE_COMPLETED",
        message=f"Triage completed with suggested goal: {assessment.suggested_goal.value}",
    )

    return {
        "triage_assessment": assessment,
        "suggested_goal": assessment.suggested_goal,
        "current_step": "triage",
        "completed_nodes": ["triage"],
        "trace_events": [event],
        "token_usage_summary": token_usage,
    }


# =============================================================================
# Node 2: Clarification Node (HITL Interrupt)
# =============================================================================

def clarification_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Human-in-the-loop checkpoint for candidate clarification when evidence
    ambiguity or unassessed reserve exceeds the 30% threshold.
    """
    existing_clarifications = state.get("user_clarifications") or {}
    if existing_clarifications:
        return {
            "current_step": "clarification",
            "completed_nodes": ["clarification"],
        }

    match_result = state.get("match_result")
    unassessed = getattr(match_result, "unassessed_reserve", 0.0) if match_result else 0.0
    user_career_obj = state.get("user_career_objective")
    jd_profile = state.get("jd_profile")
    target_role = (jd_profile.job_title or "Target Role").strip() if jd_profile else "Target Role"

    is_divergent = False
    if user_career_obj and jd_profile:
        import re
        target_words = set(re.findall(r'\b[a-z]{3,}\b', target_role.lower()))
        obj_words = set(re.findall(r'\b[a-z]{3,}\b', user_career_obj.lower()))
        stop_words = {"engineer", "developer", "senior", "lead", "associate", "specialist", "role", "position", "analyst", "advisor"}
        meaningful_target = target_words - stop_words
        meaningful_obj = obj_words - stop_words
        if meaningful_target and meaningful_obj and not (meaningful_target & meaningful_obj):
            is_divergent = True

    if is_divergent:
        prompt_data = {
            "type": "clarification_required",
            "stage": "clarification",
            "divergence_conflict": True,
            "user_career_objective": user_career_obj,
            "target_job_title": target_role,
            "message": (
                f"⚠️ Career Objective Divergence Detected: Your specified Career Objective is '{user_career_obj}', "
                f"while the target Job Description evaluated is '{target_role}'. "
                f"Please confirm whether to orient specialist roadmaps and interview preparation "
                f"toward your explicit Career Objective ({user_career_obj}) or the target JD ({target_role})."
            ),
            "options": [
                f"Focus on Career Objective: {user_career_obj}",
                f"Focus on Target JD: {target_role}",
                "Dual-Track Preparation (Both)",
            ],
        }
    else:
        prompt_data = {
            "type": "clarification_required",
            "stage": "clarification",
            "unassessed_reserve": unassessed,
            "message": (
                f"Candidate evaluation exhibits {unassessed:.1f}% unassessed reserve. "
                "Please provide clarification on target skills or role focus before proceeding."
            ),
            "options": ["Proceed with existing evidence", "Provide specialized technical focus"],
        }

    # Real LangGraph HITL interrupt
    user_input = interrupt(prompt_data)

    if isinstance(user_input, dict):
        clarification_dict = user_input
    else:
        clarification_dict = {"candidate_clarification": str(user_input)}

    event = _make_trace_event(
        node_name="clarification",
        event_type="HITL_RESUMED",
        message="Candidate clarification received via LangGraph interrupt.",
    )

    return {
        "user_clarifications": clarification_dict,
        "current_step": "clarification",
        "completed_nodes": ["clarification"],
        "trace_events": [event],
    }


# =============================================================================
# Node 3: Goal Selection Node (HITL Interrupt)
# =============================================================================

def goal_selection_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Human-in-the-loop checkpoint to explicitly capture user_goal.
    Never allows an LLM to choose or override the user's goal.
    """
    current_goal = state.get("user_goal")
    if current_goal is not None and isinstance(current_goal, UserWorkflowGoal):
        return {
            "current_step": "goal_selection",
            "completed_nodes": ["goal_selection"],
        }

    suggested = state.get("suggested_goal") or UserWorkflowGoal.FULL_PREPARATION
    suggested_str = suggested.value if hasattr(suggested, "value") else str(suggested)

    allowed_goals = [g.value for g in UserWorkflowGoal]

    prompt_data = {
        "type": "goal_selection_required",
        "stage": "goal_selection",
        "suggested_goal": suggested_str,
        "allowed_goals": allowed_goals,
        "message": f"Select workflow goal. Recommended based on triage: {suggested_str}",
    }

    # Real LangGraph HITL interrupt
    user_choice = interrupt(prompt_data)

    # Validate user choice
    selected_val = str(user_choice).strip()
    if selected_val not in allowed_goals:
        raise ValueError(
            f"Invalid workflow goal '{selected_val}'. Permitted goals are: {allowed_goals}"
        )

    selected_goal = UserWorkflowGoal(selected_val)

    event = _make_trace_event(
        node_name="goal_selection",
        event_type="HITL_RESUMED",
        message=f"User explicitly selected goal: {selected_goal.value}",
    )

    return {
        "user_goal": selected_goal,
        "current_step": "goal_selection",
        "completed_nodes": ["goal_selection"],
        "trace_events": [event],
    }


# =============================================================================
# Node 4: Career Remediation Node
# =============================================================================

def remediation_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Executes CareerRemediationAgent to synthesize phased upskilling roadmap
    and portfolio project blueprints addressing employer gaps.
    """
    from agents.remediation_agent import CareerRemediationAgent
    agent = CareerRemediationAgent(engine=_get_agent_engine())
    plan, telemetry = agent.run(
        match_result=state["match_result"],
        profile=state["resume_profile"],
        jd_profile=state.get("jd_profile"),
        user_career_objective=state.get("user_career_objective"),
    )

    tokens = telemetry.get("token_usage") or {}
    token_usage = {
        "prompt_tokens": tokens.get("prompt_tokens", 0) or 0,
        "completion_tokens": tokens.get("completion_tokens", 0) or 0,
        "total_tokens": tokens.get("total_tokens", 0) or 0,
    }

    event = _make_trace_event(
        node_name="remediation",
        event_type="NODE_COMPLETED",
        message=f"Remediation plan generated with {len(plan.milestones)} milestones.",
    )

    return {
        "remediation_plan": plan,
        "current_step": "remediation",
        "completed_nodes": ["remediation"],
        "trace_events": [event],
        "token_usage_summary": token_usage,
    }


# =============================================================================
# Node 5: Resume Optimization Node
# =============================================================================

def resume_optimization_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Executes ResumeOptimizationAgent to produce evidence-grounded STAR bullet points
    preserving employer metadata and verifying candidate skill whitelist.
    """
    from agents.resume_agent import ResumeOptimizationAgent
    agent = ResumeOptimizationAgent(engine=_get_agent_engine())
    report, telemetry = agent.run(
        profile=state["resume_profile"],
        match_result=state["match_result"],
        jd_profile=state.get("jd_profile"),
    )

    tokens = telemetry.get("token_usage") or {}
    token_usage = {
        "prompt_tokens": tokens.get("prompt_tokens", 0) or 0,
        "completion_tokens": tokens.get("completion_tokens", 0) or 0,
        "total_tokens": tokens.get("total_tokens", 0) or 0,
    }

    event = _make_trace_event(
        node_name="resume_optimization",
        event_type="NODE_COMPLETED",
        message=f"Resume optimization generated with {len(report.optimized_bullets)} STAR bullets.",
    )

    return {
        "resume_optimization": report,
        "current_step": "resume_optimization",
        "completed_nodes": ["resume_optimization"],
        "trace_events": [event],
        "token_usage_summary": token_usage,
    }


# =============================================================================
# Node 6: Interview Preparation Node
# =============================================================================

def interview_prep_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Executes InterviewPreparationAgent to produce role-specific interview kits
    grounded in candidate evidence.
    """
    from agents.interview_agent import InterviewPreparationAgent
    agent = InterviewPreparationAgent(engine=_get_agent_engine())
    kit, telemetry = agent.run(
        match_result=state["match_result"],
        profile=state["resume_profile"],
        jd_profile=state.get("jd_profile"),
        user_career_objective=state.get("user_career_objective"),
    )

    tokens = telemetry.get("token_usage") or {}
    token_usage = {
        "prompt_tokens": tokens.get("prompt_tokens", 0) or 0,
        "completion_tokens": tokens.get("completion_tokens", 0) or 0,
        "total_tokens": tokens.get("total_tokens", 0) or 0,
    }

    total_q = (
        len(kit.technical_deep_dive_questions)
        + len(kit.gap_mitigation_questions)
        + len(kit.metric_challenge_questions)
    )

    event = _make_trace_event(
        node_name="interview_prep",
        event_type="NODE_COMPLETED",
        message=f"Interview preparation kit generated with {total_q} questions.",
    )

    return {
        "interview_kit": kit,
        "current_step": "interview_prep",
        "completed_nodes": ["interview_prep"],
        "trace_events": [event],
        "token_usage_summary": token_usage,
    }


# =============================================================================
# Node 7: Deterministic Safety Audit Node
# =============================================================================

def deterministic_audit_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Executes deterministic verification across all generated deliverables:
    - Verifies score immutability and SHA-256 integrity against baseline snapshot.
    - Validates candidate skill grounding, metric disclosures, PII leakage, and evidence resolution.
    - Runs conservative prose skill-claim checks on optimized bullets.
    """
    validator = DeterministicWorkflowValidator()
    reports: list[DeterministicValidationReport] = []

    # 1. Authoritative Score & Snapshot Integrity Check
    baseline = state.get("baseline_match_snapshot")
    match_result = state.get("match_result")
    expected_sha256 = state.get("baseline_sha256")

    if baseline is not None and match_result is not None:
        verify_workflow_score_immutability(
            baseline=baseline,
            candidate=match_result,
            expected_sha256=expected_sha256,
        )

    # 2. Audit Resume Optimization if present
    resume_opt = state.get("resume_optimization")
    if resume_opt is not None:
        rep = validator.validate_resume_optimization(
            report=resume_opt,
            profile=state["resume_profile"],
            jd_profile=state.get("jd_profile"),
        )
        reports.append(rep)

    # 3. Audit Remediation Plan if present
    remediation = state.get("remediation_plan")
    if remediation is not None:
        rep = validator.validate_remediation_plan(
            plan=remediation,
            profile=state["resume_profile"],
            jd_profile=state.get("jd_profile"),
        )
        reports.append(rep)

    # 4. Audit Interview Kit if present
    interview_kit = state.get("interview_kit")
    if interview_kit is not None:
        rep = validator.validate_interview_kit(
            kit=interview_kit,
            profile=state["resume_profile"],
            jd_profile=state.get("jd_profile"),
        )
        reports.append(rep)

    # Check overall validity
    audit_passed = all(r.is_valid for r in reports) if reports else True

    event = _make_trace_event(
        node_name="deterministic_audit",
        event_type="NODE_COMPLETED",
        message=f"Deterministic audit completed. Result: {'PASSED' if audit_passed else 'FAILED'}",
    )

    return {
        "audit_passed": audit_passed,
        "validation_reports": reports,
        "current_step": "deterministic_audit",
        "completed_nodes": ["deterministic_audit"],
        "trace_events": [event],
    }


# =============================================================================
# Node 8: Bounded Correction Node
# =============================================================================

def correction_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Bounded correction node: Permits exactly 1 correction cycle when audit fails.
    Replaces non-compliant deliverables with deterministic rule-based fallbacks.
    Never removes evidence constraints or causes infinite loops.
    """
    current_attempts = state.get("correction_attempt_count", 0)
    new_attempts = current_attempts + 1

    updates: dict[str, Any] = {
        "correction_attempt_count": new_attempts,
        "current_step": "correction",
        "completed_nodes": ["correction"],
    }

    # Inspect validation reports and apply fallback to failing deliverables
    reports = state.get("validation_reports") or []
    profile = state["resume_profile"]
    match_result = state["match_result"]

    from agents.interview_agent import InterviewPreparationAgent
    from agents.remediation_agent import CareerRemediationAgent
    from agents.resume_agent import ResumeOptimizationAgent

    for rep in reports:
        if not rep.is_valid:
            if rep.target_deliverable_type == "ResumeOptimizationReport":
                updates["resume_optimization"] = ResumeOptimizationAgent.synthesize_deterministic_fallback(
                    profile=profile,
                    match_result=match_result,
                    failure_reason="Audit correction: synthesized deterministic fallback",
                )
            elif rep.target_deliverable_type == "CareerRemediationPlan":
                updates["remediation_plan"] = CareerRemediationAgent.synthesize_deterministic_fallback(
                    match_result=match_result,
                    failure_reason="Audit correction: synthesized deterministic fallback",
                )
            elif rep.target_deliverable_type == "InterviewPreparationKit":
                updates["interview_kit"] = InterviewPreparationAgent.synthesize_deterministic_fallback(
                    match_result=match_result,
                    profile=profile,
                    failure_reason="Audit correction: synthesized deterministic fallback",
                )

    event = _make_trace_event(
        node_name="correction",
        event_type="CORRECTION_EXECUTED",
        message=f"Executed bounded correction cycle {new_attempts}/1.",
    )
    updates["trace_events"] = [event]

    return updates


# =============================================================================
# Node 9: Final Approval Node (HITL Interrupt)
# =============================================================================

def final_approval_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Human-in-the-loop checkpoint requiring explicit user review and approval
    before final report synthesis and export.
    Cannot be bypassed.
    """
    decision = state.get("final_approval_decision")
    if decision in {"APPROVE", "REQUEST_CHANGES", "REJECT"}:
        return {
            "current_step": "final_approval",
            "completed_nodes": ["final_approval"],
        }

    audit_passed = state.get("audit_passed", False)
    prompt_data = {
        "type": "final_approval_required",
        "stage": "final_approval",
        "audit_passed": audit_passed,
        "deliverables_summary": {
            "has_triage": state.get("triage_assessment") is not None,
            "has_remediation": state.get("remediation_plan") is not None,
            "has_resume_opt": state.get("resume_optimization") is not None,
            "has_interview_kit": state.get("interview_kit") is not None,
        },
        "message": (
            "Review generated deliverables. Deterministic audit: "
            f"{'PASSED' if audit_passed else 'WARNINGS PRESENT'}. Explicit approval required."
        ),
        "options": ["APPROVE", "REQUEST_CHANGES", "REJECT"],
    }

    # Real LangGraph HITL interrupt
    user_decision = interrupt(prompt_data)

    decision_str = str(user_decision).strip().upper()
    if decision_str not in {"APPROVE", "REQUEST_CHANGES", "REJECT"}:
        raise ValueError(
            f"Invalid approval decision '{decision_str}'. Must be one of APPROVE, REQUEST_CHANGES, REJECT."
        )

    event = _make_trace_event(
        node_name="final_approval",
        event_type="HITL_RESUMED",
        message=f"User finalized deliverable review with decision: {decision_str}",
    )

    return {
        "final_approval_decision": decision_str,
        "current_step": "final_approval",
        "completed_nodes": ["final_approval"],
        "trace_events": [event],
    }


# =============================================================================
# Node 10: Synthesis & Export Node
# =============================================================================

def synthesis_node(state: CareerWorkflowState) -> dict[str, Any]:
    """
    Generates consolidated career intelligence report preserving Phase 4
    deterministic results, specialist deliverables, audit status, and telemetry.
    """
    match_result = state["match_result"]
    triage = state.get("triage_assessment")
    remediation = state.get("remediation_plan")
    resume_opt = state.get("resume_optimization")
    interview_kit = state.get("interview_kit")

    consolidated = {
        "workflow_id": state.get("workflow_id", "workflow_default"),
        "user_goal": state.get("user_goal").value if state.get("user_goal") else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_compatibility": {
            "overall_score": match_result.overall_score,
            "fit_classification": match_result.fit_classification.value,
            "active_evaluation_coverage": match_result.active_evaluation_coverage,
            "sha256_digest": state.get("baseline_sha256"),
            "matched_skills": [s.canonical_name for s in match_result.matched_skills],
            "missing_evidence": [g.skill_name for g in match_result.missing_evidence],
        },
        "triage_assessment": triage.model_dump(mode="json") if triage else None,
        "career_remediation": remediation.model_dump(mode="json") if remediation else None,
        "resume_optimization": resume_opt.model_dump(mode="json") if resume_opt else None,
        "interview_preparation": interview_kit.model_dump(mode="json") if interview_kit else None,
        "audit_status": {
            "audit_passed": state.get("audit_passed", False),
            "reports_count": len(state.get("validation_reports") or []),
            "correction_attempts": state.get("correction_attempt_count", 0),
        },
        "telemetry": {
            "token_usage": state.get("token_usage_summary", {}),
            "completed_nodes": state.get("completed_nodes", []),
        },
    }

    event = _make_trace_event(
        node_name="synthesis",
        event_type="NODE_COMPLETED",
        message="Consolidated career intelligence report assembled successfully.",
    )

    return {
        "consolidated_report": consolidated,
        "current_step": "synthesis",
        "completed_nodes": ["synthesis"],
        "trace_events": [event],
    }

