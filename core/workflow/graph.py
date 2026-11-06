"""
LangGraph workflow graph assembly and compilation.
Constructs official StateGraph with official MemorySaver checkpointer,
typed conditional edges, and strict Human-in-the-Loop interrupt checkpoints.
"""
from typing import Any, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.workflow.nodes import (
    clarification_node,
    correction_node,
    deterministic_audit_node,
    final_approval_node,
    goal_selection_node,
    interview_prep_node,
    remediation_node,
    resume_optimization_node,
    synthesis_node,
    triage_node,
)
from core.workflow.router import (
    route_after_audit,
    route_after_clarification,
    route_after_correction,
    route_after_final_approval,
    route_after_goal_selection,
    route_after_interview_prep,
    route_after_remediation,
    route_after_resume_optimization,
    route_after_triage,
)
from core.workflow.state import CareerWorkflowState


def build_career_workflow_graph(checkpointer: Optional[Any] = None) -> Any:
    """
    Build and compile the official LangGraph StateGraph for the
    Career Intelligence Agentic Workflow.
    """
    builder = StateGraph(CareerWorkflowState)

    # 1. Register Nodes
    builder.add_node("triage", triage_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("goal_selection", goal_selection_node)
    builder.add_node("remediation", remediation_node)
    builder.add_node("resume_optimization", resume_optimization_node)
    builder.add_node("interview_prep", interview_prep_node)
    builder.add_node("deterministic_audit", deterministic_audit_node)
    builder.add_node("correction", correction_node)
    builder.add_node("final_approval", final_approval_node)
    builder.add_node("synthesis", synthesis_node)

    # 2. Entry Edge
    builder.add_edge(START, "triage")

    # 3. Conditional Routing after Triage (Clarification check)
    builder.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "clarification": "clarification",
            "goal_selection": "goal_selection",
        },
    )

    # 4. Clarification to Goal Selection
    builder.add_edge("clarification", "goal_selection")

    # 5. Conditional Routing after Goal Selection
    builder.add_conditional_edges(
        "goal_selection",
        route_after_goal_selection,
        {
            "remediation": "remediation",
            "resume_optimization": "resume_optimization",
            "interview_prep": "interview_prep",
            "goal_selection": "goal_selection",
        },
    )

    # 6. Specialist Transitions
    builder.add_conditional_edges(
        "remediation",
        route_after_remediation,
        {
            "resume_optimization": "resume_optimization",
            "deterministic_audit": "deterministic_audit",
        },
    )

    builder.add_conditional_edges(
        "resume_optimization",
        route_after_resume_optimization,
        {
            "interview_prep": "interview_prep",
            "deterministic_audit": "deterministic_audit",
        },
    )

    builder.add_edge("interview_prep", "deterministic_audit")

    # 7. Audit & Bounded Correction Cycle
    builder.add_conditional_edges(
        "deterministic_audit",
        route_after_audit,
        {
            "final_approval": "final_approval",
            "correction": "correction",
        },
    )

    builder.add_edge("correction", "deterministic_audit")

    # 8. Final Approval & Synthesis
    builder.add_conditional_edges(
        "final_approval",
        route_after_final_approval,
        {
            "synthesis": "synthesis",
            "__end__": END,
        },
    )

    builder.add_edge("synthesis", END)

    # 9. Compilation with checkpointer
    cp = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=cp)

