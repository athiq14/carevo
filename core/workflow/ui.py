"""
Streamlit UI components for Phase 5 Agentic Career Intelligence.
Renders interactive multi-agent progress, Human-in-the-Loop checkpoints,
deliverable inspection tabs, deterministic validation audits, final approval,
and privacy-sanitized consolidated report export.
"""
import json
from typing import Any, Optional
import streamlit as st

from core.workflow.coordinator import WorkflowExecutionCoordinator
from core.workflow.validators import (
    EvidenceReferenceResolver,
    ensure_canonical_job_match_result,
)
from models.matching import JobDescriptionProfile, JobMatchResult
from models.resume_profile import ResumeProfile
from models.workflow import MetricDisclosureStatus, UserWorkflowGoal


def get_or_create_coordinator() -> WorkflowExecutionCoordinator:
    """Retrieve or initialize WorkflowExecutionCoordinator in Streamlit session state."""
    if "workflow_coordinator" not in st.session_state:
        st.session_state["workflow_coordinator"] = WorkflowExecutionCoordinator()
    return st.session_state["workflow_coordinator"]


def _safe_resume_workflow(coordinator: WorkflowExecutionCoordinator, thread_id: str, resume_value: Any) -> bool:
    """Safely resumes workflow execution with graceful error containment."""
    try:
        res_state, next_intrs = coordinator.resume_workflow(
            thread_id=thread_id,
            resume_value=resume_value,
        )
        st.session_state["workflow_state"] = res_state
        st.session_state["workflow_interrupts"] = next_intrs
        return True
    except Exception as err:
        st.error(f"❌ Workflow Execution Error: Encountered an unexpected error during agent orchestration ({err}).")
        with st.expander("🛠️ Technical Diagnostics", expanded=False):
            st.code(str(err))
        return False


def render_agentic_workflow_ui(
    match_result: Any,
    resume_profile: ResumeProfile,
    jd_profile: Optional[JobDescriptionProfile] = None,
    key_prefix: str = "agentic_wf",
) -> None:
    """
    Render the integrated Phase 5 Agentic Workflow UI section.
    Supports unique, collision-free Streamlit keys across multiple tab renderings.
    """
    match_result = ensure_canonical_job_match_result(match_result)
    st.divider()
    st.header("🤖 CAREVO Copilot (Phase 5)")
    st.caption(
        "Autonomous multi-agent LangGraph orchestration with Human-in-the-Loop checkpoints, "
        "evidence-grounded specialist reasoning, and deterministic safety auditing."
    )

    coordinator = get_or_create_coordinator()
    thread_id = st.session_state.get("active_workflow_thread_id")

    # Workflow Controls Header
    col_ctrl1, col_ctrl2 = st.columns([4, 1])
    with col_ctrl2:
        if st.button(
            "🔄 Reset Workflow",
            help="Clear active workflow session and restart co-pilot.",
            key=f"{key_prefix}_reset_workflow_btn",
        ):
            if thread_id:
                coordinator.reset_workflow(thread_id)
            for k in [
                "active_workflow_thread_id",
                "workflow_state",
                "workflow_interrupts",
                "workflow_stage_step",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

    # 1. Workflow Not Yet Started
    if not thread_id or "workflow_state" not in st.session_state:
        st.info("💡 Deterministic match evaluation complete. Launch CAREVO Copilot to generate personalized career deliverables.")
        
        goal_options = {
            "FULL_PREPARATION": "🌟 Full Preparation (Remediation + Resume STAR Bullets + Interview Drill)",
            "GAP_REMEDIATION": "📈 Gap Remediation (Phased Upskilling Roadmap & Portfolio Projects)",
            "RESUME_OPTIMIZATION": "📄 Resume Optimization (STAR Reframing & ATS Keyword Grounding)",
            "INTERVIEW_DRILL": "🎙️ Interview Drill (Technical Deep Dives & Metric Challenge Questions)",
        }

        user_career_obj_input = st.text_input(
            "🎯 Specific User Career Objective (Optional — specify if pursuing a distinct goal, e.g. Insurance Advisor):",
            value=st.session_state.get("advisor_career_obj_input", ""),
            placeholder="e.g. Insurance Advisor, AI Builder, Full-Stack Engineer",
            help="If your career objective differs from the target JD role, specify it here to guide specialist agents.",
            key=f"{key_prefix}_explicit_career_obj_input",
        )
        
        selected_goal_key = st.radio(
            "Select Initial Target Workflow Goal (Optional — Triage agent can also diagnose and recommend):",
            options=list(goal_options.keys()),
            format_func=lambda k: goal_options[k],
            index=0,
            key=f"{key_prefix}_initial_goal_radio",
        )

        col_start, _ = st.columns([2, 3])
        with col_start:
            if st.button(
                "🚀 Launch CAREVO Copilot",
                type="primary",
                key=f"{key_prefix}_launch_workflow_btn",
            ):
                with st.spinner("Initializing LangGraph workflow, creating immutable baseline, and executing Triage Agent..."):
                    try:
                        new_thread_id = f"career_wf_{id(match_result)}_{hash(resume_profile.filename) % 100000}"
                        initial_goal = UserWorkflowGoal(selected_goal_key) if selected_goal_key else None
                        state, interrupts = coordinator.start_workflow(
                            match_result=match_result,
                            resume_profile=resume_profile,
                            jd_profile=jd_profile,
                            thread_id=new_thread_id,
                            user_goal=initial_goal,
                            user_career_objective=user_career_obj_input.strip() or None,
                        )
                        st.session_state["active_workflow_thread_id"] = new_thread_id
                        st.session_state["workflow_state"] = state
                        st.session_state["workflow_interrupts"] = interrupts
                        st.rerun()
                    except Exception as err:
                        st.error(f"❌ Workflow Startup Error: Unable to initialize agentic workflow ({err}).")
                        with st.expander("🛠️ Technical Diagnostics", expanded=False):
                            st.code(str(err))
        return

    # 2. Workflow Active — Inspect State and Interrupts
    state = st.session_state["workflow_state"]
    interrupts = st.session_state.get("workflow_interrupts", [])
    current_step = state.get("current_step", "active")
    completed_nodes = state.get("completed_nodes", [])

    # Persistent Career Objective Banner
    active_obj = state.get("user_career_objective")
    if active_obj:
        st.markdown(
            f"""
            <div style="background: rgba(139, 69, 247, 0.12); border: 1px solid rgba(139, 69, 247, 0.4); border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 13px; font-weight: 700; color: #A78BFA; text-transform: uppercase; letter-spacing: 0.05em;">EXPLICIT USER CAREER OBJECTIVE</span>
                    <div style="font-size: 19px; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{active_obj}</div>
                </div>
                <div style="font-size: 13px; color: #36D399; font-weight: 600; background: rgba(54, 211, 153, 0.15); padding: 5px 12px; border-radius: 6px;">
                    Active in Specialist Agents ✔
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Workflow Progress Bar
    all_stages = ["triage", "clarification", "goal_selection", "remediation", "resume_optimization", "interview_prep", "deterministic_audit", "final_approval", "synthesis"]
    stage_labels = {
        "triage": "1. Career Assessment",
        "clarification": "2. Clarification",
        "goal_selection": "3. Goal Selection",
        "remediation": "4. Skill Development",
        "resume_optimization": "5. Resume Improvement",
        "interview_prep": "6. Interview Preparation",
        "deterministic_audit": "7. Safety Review",
        "final_approval": "8. Your Approval",
        "synthesis": "9. Final Report",
    }

    progress_cols = st.columns(len(all_stages))
    for i, stage in enumerate(all_stages):
        with progress_cols[i]:
            if stage in completed_nodes:
                st.caption(f"🟣 {stage_labels[stage]}")
            elif stage == current_step or (interrupts and interrupts[0].get("stage") == stage):
                st.caption(f"⚡ **{stage_labels[stage]}**")
            else:
                st.caption(f"⚪ {stage_labels[stage]}")

    st.divider()

    # 3. Handle Active HITL Interrupt Checkpoints
    if interrupts:
        active_intr = interrupts[0]
        intr_type = active_intr.get("type")

        # Checkpoint A: Clarification Interrupt
        if intr_type == "clarification_required":
            is_div = active_intr.get("divergence_conflict", False)
            if is_div:
                st.warning("### ⚠️ Human-in-the-Loop Checkpoint: Career Objective Divergence Detected")
            else:
                st.warning("### ⚠️ Human-in-the-Loop Checkpoint: Candidate Clarification Required")
            st.markdown(
                f"**Reason:** {active_intr.get('message')}\n\n"
                "To maximize recommendation accuracy, please confirm your preferred direction or provide additional context."
            )
            options = active_intr.get("options", [])
            if options:
                clarification_choice = st.radio(
                    "Select Resolution Focus:",
                    options=options,
                    key=f"{key_prefix}_clarification_opt_radio",
                )
            else:
                clarification_choice = st.text_area(
                    "Candidate Clarification Input:",
                    placeholder="e.g. 'Please focus on my explicit career objective: Insurance Advisor.'",
                    key=f"{key_prefix}_clarification_input",
                )
            col_clar_btn, _ = st.columns([2, 3])
            with col_clar_btn:
                if st.button(
                    "Submit Clarification & Resume Workflow",
                    type="primary",
                    key=f"{key_prefix}_submit_clarification_btn",
                ):
                    with st.spinner("Delivering clarification and resuming LangGraph execution..."):
                        if _safe_resume_workflow(
                            coordinator=coordinator,
                            thread_id=thread_id,
                            resume_value=clarification_choice or "Proceed with verified resume profile",
                        ):
                            st.rerun()
            return

        # Checkpoint B: Goal Selection Interrupt
        elif intr_type == "goal_selection_required":
            st.info("### 🎯 Human-in-the-Loop Checkpoint: Explicit Workflow Goal Selection")
            suggested = active_intr.get("suggested_goal", "FULL_PREPARATION")
            st.markdown(f"**Triage Diagnostics Recommendation:** `{suggested}`")
            
            allowed = active_intr.get("allowed_goals", [g.value for g in UserWorkflowGoal])
            user_selected_goal = st.radio(
                "Confirm or Override Target Workflow Goal:",
                options=allowed,
                index=allowed.index(suggested) if suggested in allowed else 0,
                key=f"{key_prefix}_goal_selection_radio",
            )
            col_goal_btn, _ = st.columns([2, 3])
            with col_goal_btn:
                if st.button(
                    "Confirm Goal & Execute Specialists",
                    type="primary",
                    key=f"{key_prefix}_confirm_goal_btn",
                ):
                    with st.spinner("Executing specialist reasoning agents and deterministic audit..."):
                        if _safe_resume_workflow(
                            coordinator=coordinator,
                            thread_id=thread_id,
                            resume_value=user_selected_goal,
                        ):
                            st.rerun()
            return

        # Checkpoint C: Final Approval Interrupt (Awaiting User Review)
        elif intr_type == "final_approval_required":
            audit_passed = active_intr.get("audit_passed", False)
            st.info("### ⏳ Workflow Run Completed — Awaiting Your Authorization")
            if audit_passed:
                st.success(
                    "### ✅ Human-in-the-Loop Checkpoint: Deliverables Ready for Final Review\n\n"
                    "All generated deliverables have successfully passed deterministic grounding, "
                    "candidate skill whitelisting, metric disclosure checks, and SHA-256 score immutability verification."
                )
            else:
                st.warning(
                    "### ⚠️ Human-in-the-Loop Checkpoint: Deliverables Audit Completed with Warnings\n\n"
                    "One or more deliverables required automatic fallback correction to enforce factual grounding. "
                    "Please review the deliverables below before authorizing."
                )

            st.markdown("#### Review Generated Deliverables Below:")
            _render_deliverables_tabs(state, resume_profile=resume_profile)

            st.divider()
            st.markdown("#### Provide Final Authorization Decision:")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button(
                    "🟢 Approve & Synthesize Final Report",
                    type="primary",
                    key=f"{key_prefix}_approve_deliverables_btn",
                ):
                    with st.spinner("Finalizing report synthesis and exporting consolidated intelligence bundle..."):
                        if _safe_resume_workflow(
                            coordinator=coordinator,
                            thread_id=thread_id,
                            resume_value="APPROVE",
                        ):
                            st.rerun()
            with btn_col2:
                if st.button(
                    "🟡 Request Changes",
                    key=f"{key_prefix}_request_changes_btn",
                ):
                    with st.spinner("Recording change request..."):
                        if _safe_resume_workflow(
                            coordinator=coordinator,
                            thread_id=thread_id,
                            resume_value="REQUEST_CHANGES",
                        ):
                            st.rerun()
            with btn_col3:
                if st.button(
                    "🔴 Reject Deliverables",
                    key=f"{key_prefix}_reject_deliverables_btn",
                ):
                    with st.spinner("Rejecting deliverables..."):
                        if _safe_resume_workflow(
                            coordinator=coordinator,
                            thread_id=thread_id,
                            resume_value="REJECT",
                        ):
                            st.rerun()
            return

    # 4. Completed Workflow — Distinct States for APPROVE, REQUEST_CHANGES, and REJECT
    decision = state.get("final_approval_decision")

    if decision == "APPROVE":
        st.success("### ✅ CAREVO Copilot — Agentic Workflow Execution Completed (Approved & Synthesized)")
        st.write(f"**Final Approval Decision:** `{decision}`")
        st.markdown(
            """
            <div style="background: rgba(54, 211, 153, 0.1); border: 1px solid rgba(54, 211, 153, 0.35); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <strong style="color: #36D399; font-size: 16px;">Status: Approved</strong><br/>
                <span style="color: #F8FAFC; font-size: 15px; line-height: 1.55;">All specialist deliverables were reviewed and authorized by the candidate. The consolidated intelligence bundle has been synthesized with verified score immutability and is ready for export.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif decision == "REQUEST_CHANGES":
        st.warning("### 🟡 Workflow Run Completed — Changes Requested")
        st.write(f"**Final Approval Decision:** `{decision}` (Deliverables Not Approved)")
        st.markdown(
            """
            <div style="background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.35); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <strong style="color: #FBBF24; font-size: 16px;">Status: Changes Requested (Deliverables Not Approved)</strong><br/>
                <span style="color: #F8FAFC; font-size: 15px; line-height: 1.55;">The multi-agent workflow finished generating draft deliverables, but you requested modifications. <em>Because changes were requested, these deliverables are NOT approved</em> and final report synthesis was omitted. You may inspect the draft outputs below or click "Reset Workflow" to restart with adjusted goals.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif decision == "REJECT":
        st.error("### 🔴 Workflow Run Terminated — Deliverables Rejected")
        st.write(f"**Final Approval Decision:** `{decision}` (Unapproved)")
        st.markdown(
            """
            <div style="background: rgba(251, 113, 133, 0.1); border: 1px solid rgba(251, 113, 133, 0.35); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <strong style="color: #FB7185; font-size: 16px;">Status: Rejected (Unapproved)</strong><br/>
                <span style="color: #F8FAFC; font-size: 15px; line-height: 1.55;">The generated deliverables were explicitly rejected by the candidate. The workflow terminated without approval, and final report synthesis was aborted. Review draft outputs below or reset the workflow to restart.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("### ⚡ Workflow Run Completed")
        st.caption("Specialist deliverables generated and ready for review.")

    # Render Deliverables Tabs
    _render_deliverables_tabs(state, resume_profile=resume_profile)

    # Consolidated Report Download ONLY if APPROVED
    report = state.get("consolidated_report")
    if decision == "APPROVE" and report:
        st.divider()
        st.subheader("💾 Download Consolidated Career Intelligence Bundle")
        report_json = json.dumps(report, indent=2, default=str)
        st.download_button(
            label="📥 Download Consolidated Intelligence Report (JSON)",
            data=report_json,
            file_name=f"carevo_intelligence_report_{thread_id[:12]}.json",
            mime="application/json",
            help="Download the consolidated multi-agent report with match scores, triage diagnostics, upskilling plan, STAR bullets, and interview kit.",
            key=f"{key_prefix}_download_consolidated_report_btn",
        )
    elif decision in {"REQUEST_CHANGES", "REJECT"}:
        st.divider()
        st.info(
            f"ℹ️ **Consolidated Intelligence Bundle Restricted**: Final report synthesis and download are disabled because deliverables were marked as `{decision}`. Report synthesis is restricted strictly to approved deliverables."
        )


def _render_deliverables_tabs(state: dict[str, Any], resume_profile: Optional[ResumeProfile] = None) -> None:
    """Helper to render specialist deliverable tabs with honest provenance and side-by-side comparison."""
    triage = state.get("triage_assessment")
    remediation = state.get("remediation_plan")
    resume_opt = state.get("resume_optimization")
    interview_kit = state.get("interview_kit")
    validation_reports = state.get("validation_reports") or []
    effective_prof = resume_profile or state.get("resume_profile")

    tab_triage, tab_rem, tab_resume, tab_interview, tab_audit = st.tabs([
        "🔍 Career Assessment (Triage)",
        "📈 Skill Development Coach",
        "📄 Resume Improvement (STAR)",
        "🎙️ Interview Coach",
        "🛡️ Safety & Score Audit",
    ])

    # Tab 1: Triage Diagnostics
    with tab_triage:
        if triage:
            st.markdown(f"#### Primary Qualification Bottleneck\n> {triage.primary_bottleneck}")
            st.markdown(f"**Suggested Workflow Goal:** `{triage.suggested_goal.value}`")
            st.markdown(f"**Advisory Rationale:** {triage.advisory_rationale}")

            col_s, col_g = st.columns(2)
            with col_s:
                st.markdown("##### ⭐ Key Strengths")
                for s in triage.key_strengths_summary:
                    st.markdown(f"- {s}")
            with col_g:
                st.markdown("##### ⚠️ Critical Gaps")
                for g in triage.critical_gaps_summary:
                    st.markdown(f"- {g}")
        else:
            st.info("Triage diagnostics not available.")

    # Tab 2: Career Remediation
    with tab_rem:
        if remediation:
            st.markdown(f"#### Strategic Trajectory Overview\n{remediation.executive_summary}")
            st.caption(f"Estimated Readiness Timeline: **{remediation.estimated_readiness_timeline}**")

            st.markdown("##### 📅 Phased Upskilling Milestones")
            for m in remediation.milestones:
                with st.expander(f"📌 {m.timeframe}: Targeting {', '.join(m.target_missing_skills)}", expanded=True):
                    st.markdown("**Learning Objectives:**")
                    for obj in m.learning_objectives:
                        st.markdown(f"- {obj}")
                    st.markdown(f"**Verifiable Deliverable:** `{m.concrete_deliverable}`")

            st.markdown("##### 🏗️ Recommended Portfolio Projects")
            for p in remediation.portfolio_projects:
                with st.expander(f"🚀 Project Blueprint: {p.project_concept}", expanded=True):
                    st.markdown(f"**Architecture Overview:**\n{p.architecture_overview}")
                    st.markdown(f"**Recommended Stack:** {' '.join(f'`{t}`' for t in p.recommended_tech_stack)}")
                    st.markdown("**Production-Grade Checklist:**")
                    for item in p.production_grade_checklist:
                        st.markdown(f"- [x] {item}")
        else:
            st.info("Career remediation plan not executed for this workflow goal.")

    # Tab 3: Resume Improvement (STAR)
    with tab_resume:
        if resume_opt:
            st.markdown(f"#### Target Role Alignment Summary\n{resume_opt.target_role_alignment_summary}")

            st.markdown("##### 📝 Grounded STAR Bullet Points")
            st.caption(
                "Evidence Honesty Notice: Every bullet is reframed strictly from candidate-reported evidence. "
                "Bullets are source-linked to parsed resume records. Independent third-party audit is never falsely claimed."
            )

            for idx, b in enumerate(resume_opt.optimized_bullets, start=1):
                # Distinguish provenance tiers honestly
                if "user_clarifications" in b.source_field_path:
                    prov_badge = "👤 Candidate-Confirmed"
                    prov_desc = "Explicitly clarified and confirmed by candidate during workflow checkpoint"
                elif hasattr(b, "provenance_label") and b.provenance_label:
                    prov_badge = f"📄 {b.provenance_label}"
                    prov_desc = f"Source-linked to candidate resume evidence ({b.source_entity_id})"
                else:
                    prov_badge = "📄 Rewritten from candidate-provided evidence"
                    prov_desc = f"Source-linked to candidate resume evidence ({b.source_entity_id})"

                # Metric disclosure badge
                if b.metric_disclosure.value == MetricDisclosureStatus.SELF_REPORTED_METRIC.value:
                    metric_badge = "🟡 Self-Reported Metric (Candidate Claim, Not Externally Audited)"
                else:
                    metric_badge = "ℹ️ Qualitative Narrative (Candidate-Reported Evidence)"

                # Resolve original resume evidence for side-by-side comparison
                orig_text = None
                if effective_prof:
                    is_res, _, val = EvidenceReferenceResolver.resolve_reference(
                        b.source_entity_id, b.source_field_path, effective_prof
                    )
                    if is_res and val:
                        orig_text = str(val)

                with st.expander(f"Bullet #{idx}: {prov_badge} [{b.source_entity_id}]", expanded=True):
                    struct_label = getattr(b, "bullet_structure", "Action & Scope (Self-Reported Responsibility)")
                    st.caption(f"**Structure:** `{struct_label}` | **Provenance:** {prov_desc} | **Metric Status:** {metric_badge}")

                    # Side-by-side comparison between original evidence and improved STAR bullet
                    col_orig, col_imp = st.columns(2)
                    with col_orig:
                        st.markdown("<div style='font-size: 14px; font-weight: 700; color: #A1A1B5; margin-bottom: 6px;'>📄 Original Resume Evidence:</div>", unsafe_allow_html=True)
                        if orig_text:
                            st.markdown(
                                f"<div style='background: rgba(17, 20, 38, 0.9); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 14px; font-size: 15px; line-height: 1.55; color: #F8FAFC; min-height: 70px;'>\"{orig_text}\"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div style='background: rgba(17, 20, 38, 0.9); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; padding: 14px; font-size: 15px; line-height: 1.55; color: #A1A1B5; min-height: 70px;'>Record: {b.source_entity_id} ({b.source_field_path})</div>",
                                unsafe_allow_html=True,
                            )

                    with col_imp:
                        st.markdown("<div style='font-size: 14px; font-weight: 700; color: #8B45F7; margin-bottom: 6px;'>✨ Improved STAR Bullet:</div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='background: rgba(139, 69, 247, 0.1); border: 1px solid rgba(139, 69, 247, 0.4); border-radius: 8px; padding: 14px; font-size: 15px; line-height: 1.55; color: #F8FAFC; font-weight: 500; min-height: 70px;'>\"{b.improved_star_bullet}\"</div>",
                            unsafe_allow_html=True,
                        )

                    st.caption(f"**Highlighted Skills:** {', '.join(f'`{s}`' for s in b.highlighted_skills) if b.highlighted_skills else 'None'}")
                    st.caption(f"**Alignment Rationale:** {b.alignment_rationale} | **Field Path:** `{b.source_field_path}`")

            if resume_opt.ats_keyword_density_feedback:
                st.markdown("##### 📊 ATS Keyword & Structural Recommendations")
                for rec in resume_opt.ats_keyword_density_feedback:
                    st.markdown(f"- {rec}")
        else:
            st.info("Resume optimization not executed for this workflow goal.")

    # Tab 4: Interview Kit
    with tab_interview:
        if interview_kit:
            st.markdown(f"#### Interview Preparation Overview\n{interview_kit.role_context_overview}")

            st.markdown("##### 🔬 Technical Deep-Dive Inquiries")
            for q in interview_kit.technical_deep_dive_questions:
                with st.expander(f"Question: {q.question_text}", expanded=True):
                    st.caption(f"Interviewer Intent: *{q.interviewer_intent}* | Target: `{q.targeted_skill_or_gap}`")
                    st.markdown("**Suggested Talking Points:**")
                    for tp in q.suggested_talking_points:
                        st.markdown(f"- {tp}")

            st.markdown("##### 🛡️ Gap Mitigation Coaching Questions")
            for q in interview_kit.gap_mitigation_questions:
                with st.expander(f"Probe: {q.question_text}", expanded=True):
                    st.caption(f"Interviewer Intent: *{q.interviewer_intent}* | Target Gap: `{q.targeted_skill_or_gap}`")
                    st.markdown("**Coached Talking Points:**")
                    for tp in q.suggested_talking_points:
                        st.markdown(f"- {tp}")

            st.markdown("##### 🙋 Recommended Questions for Candidate to Ask")
            for q_text in interview_kit.recommended_questions_for_interviewer:
                st.markdown(f"- *\"{q_text}\"*")
        else:
            st.info("Interview preparation kit not executed for this workflow goal.")

    # Tab 5: Safety & Score Audit
    with tab_audit:
        st.markdown("#### 🛡️ Deterministic Safety & Score Integrity Audit")
        st.write(f"- **Audit Status:** `{'🟢 PASSED' if state.get('audit_passed') else '🔴 WARNINGS DETECTED'}`")
        st.write(f"- **Correction Cycles Executed:** `{state.get('correction_attempt_count', 0)} / 1 max`")
        st.write(f"- **Cryptographic SHA-256 Digest Anchor:** `{state.get('baseline_sha256')}`")
        st.write("- **Phase 4 Authoritative Score Modification:** `0 (Strictly Blocked & Verified)`")

        if validation_reports:
            for rep in validation_reports:
                st.markdown(f"##### Report: `{rep.target_deliverable_type}` ({'🟢 Valid' if rep.is_valid else '🔴 Invalid'})")
                if rep.violations:
                    for v in rep.violations:
                        st.warning(f"[{v.severity.value}] {v.violation_type.value}: {v.message} (Target: `{v.target_field}`)")
                else:
                    st.caption("Zero violations detected.")
        else:
            st.caption("No validation reports registered.")

