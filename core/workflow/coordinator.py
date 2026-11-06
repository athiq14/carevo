"""
Workflow Execution Coordinator for Phase 5 Agentic Career Intelligence.
Coordinates official LangGraph execution, thread management, Human-in-the-Loop interrupts,
and enforces out-of-band SHA-256 cryptographic score integrity verification.
"""
from typing import Any, Optional, Union
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from core.matching.exceptions import ScoreImmutabilityError
from core.matching.pipeline import PipelineMatchResult
from core.workflow.graph import build_career_workflow_graph
from core.workflow.state import CareerWorkflowState
from core.workflow.validators import (
    create_baseline_snapshot,
    ensure_canonical_job_match_result,
    intercept_node_update,
    verify_workflow_score_immutability,
)
from models.matching import JobDescriptionProfile, JobMatchResult
from models.resume_profile import ResumeProfile
from models.workflow import AuthoritativeMatchSnapshot, UserWorkflowGoal


class WorkflowExecutionCoordinator:
    """
    Central execution service coordinating LangGraph execution threads,
    handling pause/resume HITL states, and guaranteeing authoritative score immutability.
    """

    def __init__(self, checkpointer: Optional[Any] = None) -> None:
        self._checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self._graph = build_career_workflow_graph(checkpointer=self._checkpointer)
        # Authoritative SHA-256 anchors maintained strictly OUTSIDE mutable graph state
        self._integrity_anchors: dict[str, str] = {}
        # Authoritative immutable snapshots maintained OUTSIDE mutable graph state
        self._baselines: dict[str, AuthoritativeMatchSnapshot] = {}

    @property
    def checkpointer(self) -> Any:
        """Access underlying checkpointer."""
        return self._checkpointer

    @property
    def graph(self) -> Any:
        """Access compiled LangGraph instance."""
        return self._graph

    def start_workflow(
        self,
        match_result: Union[JobMatchResult, PipelineMatchResult],
        resume_profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
        thread_id: Optional[str] = None,
        user_goal: Optional[UserWorkflowGoal] = None,
        user_career_objective: Optional[str] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Initialize and execute the career intelligence workflow up to the first HITL checkpoint.

        Args:
            match_result: Deterministic JobMatchResult or PipelineMatchResult.
            resume_profile: Parsed candidate ResumeProfile.
            jd_profile: Target job description profile.
            thread_id: Optional unique thread identifier (generated if omitted).
            user_goal: Optional pre-selected workflow goal.

        Returns:
            (state_dict, pending_interrupts)
        """
        # Extract underlying JobMatchResult if PipelineMatchResult passed
        if isinstance(match_result, PipelineMatchResult) or (
            getattr(match_result, "__class__", None) is not None and match_result.__class__.__name__ == "PipelineMatchResult"
        ):
            actual_match = match_result.match_result
        else:
            actual_match = match_result

        actual_match = ensure_canonical_job_match_result(actual_match)

        active_thread_id = thread_id or f"workflow_thread_{uuid.uuid4().hex[:12]}"

        # 1. Capture independent immutable baseline and compute SHA-256 anchor
        snapshot = create_baseline_snapshot(actual_match)
        expected_sha256 = snapshot.sha256_digest

        # Store anchor outside mutable graph state
        self._integrity_anchors[active_thread_id] = expected_sha256
        self._baselines[active_thread_id] = snapshot

        # 2. Build initial state dictionary
        initial_state: CareerWorkflowState = {
            "workflow_id": active_thread_id,
            "match_result": actual_match,
            "baseline_match_snapshot": snapshot,
            "baseline_sha256": expected_sha256,
            "resume_profile": resume_profile,
            "jd_profile": jd_profile,
            "suggested_goal": None,
            "user_goal": user_goal,
            "user_career_objective": user_career_objective.strip() if user_career_objective and user_career_objective.strip() else None,
            "active_path": [],
            "current_step": "initialization",
            "completed_nodes": [],
            "provenance_records": [],
            "user_clarifications": {},
            "triage_assessment": None,
            "remediation_plan": None,
            "resume_optimization": None,
            "interview_kit": None,
            "validation_reports": [],
            "correction_attempt_count": 0,
            "audit_passed": False,
            "trace_events": [],
            "token_usage_summary": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "workflow_errors": [],
            "final_approval_decision": None,
            "consolidated_report": None,
        }

        config = {"configurable": {"thread_id": active_thread_id}}

        # 3. Execute graph until complete or paused on interrupt
        res = self._graph.invoke(initial_state, config=config)

        # 4. Out-of-band SHA-256 score integrity verification
        self._verify_thread_integrity(active_thread_id, res)

        interrupts = self._extract_interrupts(res)
        return res, interrupts

    def resume_workflow(
        self,
        thread_id: str,
        resume_value: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Resume a paused workflow thread using Command(resume=resume_value).

        Args:
            thread_id: Active thread identifier.
            resume_value: User input to deliver to the waiting interrupt.

        Returns:
            (state_dict, pending_interrupts)
        """
        if thread_id not in self._integrity_anchors:
            raise KeyError(f"Thread '{thread_id}' not found in coordinator session registry.")

        config = {"configurable": {"thread_id": thread_id}}

        # Resume using official LangGraph Command(resume=...)
        res = self._graph.invoke(Command(resume=resume_value), config=config)

        # Out-of-band SHA-256 score integrity verification
        self._verify_thread_integrity(thread_id, res)

        interrupts = self._extract_interrupts(res)
        return res, interrupts

    def get_state(self, thread_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Retrieve current state and pending interrupts for a thread without resuming execution.
        """
        if thread_id not in self._integrity_anchors:
            raise KeyError(f"Thread '{thread_id}' not found in coordinator session registry.")

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)

        state_values = dict(snapshot.values) if snapshot and snapshot.values else {}
        interrupts: list[dict[str, Any]] = []

        if snapshot and snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    for intr in task.interrupts:
                        val = getattr(intr, "value", intr)
                        interrupts.append(val if isinstance(val, dict) else {"value": val})

        return state_values, interrupts

    def reset_workflow(self, thread_id: str) -> None:
        """Clean up thread resources and integrity anchors."""
        self._integrity_anchors.pop(thread_id, None)
        self._baselines.pop(thread_id, None)

    def _verify_thread_integrity(self, thread_id: str, state_dict: dict[str, Any]) -> None:
        """
        Verify that the match_result in state_dict matches the external SHA-256 anchor.
        Raises ScoreImmutabilityError immediately if tampered with.
        """
        expected_sha256 = self._integrity_anchors.get(thread_id)
        baseline = self._baselines.get(thread_id)

        if expected_sha256 is None or baseline is None:
            raise ScoreImmutabilityError(
                f"Missing external integrity anchor for workflow thread '{thread_id}'."
            )

        candidate_result = state_dict.get("match_result")
        if candidate_result is None:
            raise ScoreImmutabilityError(
                f"Workflow state for thread '{thread_id}' lacks valid JobMatchResult."
            )
        try:
            candidate_valid = ensure_canonical_job_match_result(candidate_result)
        except TypeError as err:
            raise ScoreImmutabilityError(
                f"Workflow state for thread '{thread_id}' lacks valid JobMatchResult: {err}"
            ) from err

        verify_workflow_score_immutability(
            baseline=baseline,
            candidate=candidate_valid,
            expected_sha256=expected_sha256,
        )

    @staticmethod
    def _extract_interrupts(res: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract pending interrupt payloads from graph response dictionary."""
        interrupts: list[dict[str, Any]] = []
        raw_list = res.get("__interrupt__")
        if raw_list:
            for item in raw_list:
                val = getattr(item, "value", item)
                if isinstance(val, dict):
                    interrupts.append(val)
                else:
                    interrupts.append({"value": val})
        return interrupts

