"""
LangGraph state definitions and reducer functions for Phase 5 Agentic Workflows.
Implements official LangGraph-compatible TypedDict state with selective reducers,
guaranteeing deduplication and preserving Phase 1-4 authoritative immutability.
"""
from typing import Annotated, Any, Optional, TypedDict

from models.matching import JobDescriptionProfile, JobMatchResult
from models.resume_profile import ResumeProfile
from models.workflow import (
    AuthoritativeMatchSnapshot,
    CareerRemediationPlan,
    DeterministicValidationReport,
    EvidenceProvenanceRecord,
    InterviewPreparationKit,
    ResumeOptimizationReport,
    TriageAssessment,
    UserWorkflowGoal,
    WorkflowTraceEvent,
)


# =============================================================================
# Pure Reducer Functions (Deduplication & Controlled Accumulation)
# =============================================================================

def reduce_unique_strings(existing: list[str], new_items: list[str]) -> list[str]:
    """
    Reducer for string lists that preserves order while preventing duplicate entries.
    Ensures node execution history does not record duplicate node identifiers.
    """
    if not existing:
        return list(dict.fromkeys(new_items)) if new_items else []
    if not new_items:
        return list(existing)

    seen = set(existing)
    merged = list(existing)
    for item in new_items:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def reduce_provenance_records(
    existing: list[EvidenceProvenanceRecord],
    new_records: list[EvidenceProvenanceRecord],
) -> list[EvidenceProvenanceRecord]:
    """
    Reducer accumulating EvidenceProvenanceRecord entries with deduplication by provenance_id.
    """
    if not existing:
        return list(new_records) if new_records else []
    if not new_records:
        return list(existing)

    seen_ids = {r.provenance_id for r in existing}
    merged = list(existing)
    for rec in new_records:
        if rec.provenance_id not in seen_ids:
            seen_ids.add(rec.provenance_id)
            merged.append(rec)
    return merged


def reduce_validation_reports(
    existing: list[DeterministicValidationReport],
    new_reports: list[DeterministicValidationReport],
) -> list[DeterministicValidationReport]:
    """
    Reducer accumulating DeterministicValidationReport entries with deduplication by report_id.
    """
    if not existing:
        return list(new_reports) if new_reports else []
    if not new_reports:
        return list(existing)

    seen_ids = {r.report_id for r in existing}
    merged = list(existing)
    for rep in new_reports:
        if rep.report_id not in seen_ids:
            seen_ids.add(rep.report_id)
            merged.append(rep)
    return merged


def reduce_trace_events(
    existing: list[WorkflowTraceEvent],
    new_events: list[WorkflowTraceEvent],
) -> list[WorkflowTraceEvent]:
    """
    Reducer accumulating telemetry WorkflowTraceEvent entries with deduplication by event_id.
    """
    if not existing:
        return list(new_events) if new_events else []
    if not new_events:
        return list(existing)

    seen_ids = {e.event_id for e in existing}
    merged = list(existing)
    for evt in new_events:
        if evt.event_id not in seen_ids:
            seen_ids.add(evt.event_id)
            merged.append(evt)
    return merged


def reduce_workflow_errors(
    existing: list[dict[str, str]],
    new_errors: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Reducer accumulating error dictionaries without duplicate records.
    """
    if not existing:
        return list(new_errors) if new_errors else []
    if not new_errors:
        return list(existing)

    merged = list(existing)
    for err in new_errors:
        if err not in merged:
            merged.append(err)
    return merged


def reduce_token_usage(
    existing: dict[str, int],
    update: dict[str, int],
) -> dict[str, int]:
    """
    Reducer combining token usage counters by summing values for matching keys.
    """
    if not existing:
        return dict(update) if update else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not update:
        return dict(existing)

    merged = dict(existing)
    for key, val in update.items():
        merged[key] = merged.get(key, 0) + int(val)
    return merged


def reduce_clarifications(
    existing: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    """
    Reducer merging candidate clarification inputs from HITL interactions.
    """
    if not existing:
        return dict(update) if update else {}
    if not update:
        return dict(existing)
    return {**existing, **update}


# =============================================================================
# LangGraph TypedDict State Schema
# =============================================================================

class CareerWorkflowState(TypedDict, total=False):
    """
    Official LangGraph-compatible state dictionary for multi-agent workflows.

    Partial Update Semantics:
        Nodes return dictionaries containing only the keys they update.
        Fields marked with `Annotated[T, reducer]` merge according to their reducer.
        Fields without reducers overwrite their prior value on return.

    Authoritative Input Fields (Immutable):
        - `workflow_id`: Unique identifier for the execution thread.
        - `match_result`: Authoritative Phase 4 deterministic JobMatchResult (cannot be mutated).
        - `resume_profile`: Parsed Phase 2 candidate profile (cannot be mutated).
        - `jd_profile`: Normalized Phase 4 employer job description profile (cannot be mutated).
    """

    # 1. Authoritative Phase 1-4 Input Context (Immutable)
    workflow_id: str
    match_result: JobMatchResult
    baseline_match_snapshot: AuthoritativeMatchSnapshot
    baseline_sha256: str
    resume_profile: ResumeProfile
    jd_profile: JobDescriptionProfile

    # 2. Workflow Orchestration & Navigation
    suggested_goal: Optional[UserWorkflowGoal]
    user_goal: Optional[UserWorkflowGoal]
    user_career_objective: Optional[str]
    active_path: list[str]
    current_step: str
    completed_nodes: Annotated[list[str], reduce_unique_strings]

    # 3. Provenance & User Interaction Context
    provenance_records: Annotated[list[EvidenceProvenanceRecord], reduce_provenance_records]
    user_clarifications: Annotated[dict[str, Any], reduce_clarifications]
    final_approval_decision: Optional[str]

    # 4. Specialist Agent Deliverables (Overwritten per agent run)
    triage_assessment: Optional[TriageAssessment]
    remediation_plan: Optional[CareerRemediationPlan]
    resume_optimization: Optional[ResumeOptimizationReport]
    interview_kit: Optional[InterviewPreparationKit]
    consolidated_report: Optional[dict[str, Any]]

    # 5. Deterministic Validation & Audit Tracking
    validation_reports: Annotated[list[DeterministicValidationReport], reduce_validation_reports]
    correction_attempt_count: int
    audit_passed: bool

    # 6. Observability, Telemetry & Errors
    trace_events: Annotated[list[WorkflowTraceEvent], reduce_trace_events]
    token_usage_summary: Annotated[dict[str, int], reduce_token_usage]
    workflow_errors: Annotated[list[dict[str, str]], reduce_workflow_errors]

