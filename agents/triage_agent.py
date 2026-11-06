"""
Specialist Triage & Diagnostics Agent for Phase 5.2.
Evaluates deterministic Phase 4 match results, identifies primary qualification bottlenecks,
summarizes core strengths and critical gaps, and suggests an advisory workflow path.
Never selects or modifies the authoritative user_goal.
"""
from typing import Any, Optional
from agents.base import BaseSpecialistAgent
from core.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from core.matching.explanation_generator import scrub_pii
from models.matching import CandidateFitLevel, JobDescriptionProfile, JobMatchResult
from models.workflow import TriageAssessment, UserWorkflowGoal

TRIAGE_SYSTEM_PROMPT = """You are the Senior Technical Career Triage and Diagnostics Agent.
Your responsibility is to analyze deterministic job-match evaluation results and produce an objective diagnostic assessment.

Guidelines:
1. Objectively evaluate candidate qualification evidence, dimensional scores, verified competencies, and evidence gaps.
2. Identify the single primary bottleneck preventing optimal candidate-role alignment.
3. Suggest one advisory workflow path from the four permitted goals:
   - FULL_PREPARATION: Comprehensive roadmap spanning skill remediation, resume optimization, and interview drill.
   - GAP_REMEDIATION: Deep skill gaps requiring prioritized learning and portfolio development before application.
   - RESUME_OPTIMIZATION: Strong technical alignment where evidence presentation and STAR framing are the main bottlenecks.
   - INTERVIEW_DRILL: High technical and experience alignment; focus on technical deep dives and metric articulation.
4. Ground every strength and gap strictly in the provided evaluation data. Do not fabricate skills or achievements.
5. Provide a clear, actionable advisory rationale explaining why the suggested path yields the highest candidate ROI.
"""


class TriageAgent(BaseSpecialistAgent):
    """
    Evaluates candidate matching facts and produces a structured TriageAssessment.
    Provides robust deterministic fallback behavior when LLM is offline or encounters errors.
    """

    def run(
        self,
        match_result: JobMatchResult,
        jd_profile: Optional[JobDescriptionProfile] = None,
        max_attempts: int = 2,
    ) -> tuple[TriageAssessment, dict[str, Any]]:
        """
        Execute triage evaluation on deterministic match result.
        Returns (TriageAssessment, telemetry_metadata).
        """
        try:
            payload = self._build_sanitized_triage_payload(match_result, jd_profile)
            user_content = (
                "Please analyze the following deterministic career matching facts and synthesize a diagnostic assessment:\n\n"
                f"{self._format_safe_envelope('deterministic_matching_facts', payload)}"
            )

            request = LLMRequest(
                messages=[
                    LLMMessage(role=MessageRole.SYSTEM, content=TRIAGE_SYSTEM_PROMPT),
                    LLMMessage(role=MessageRole.USER, content=user_content),
                ],
                generation_config=LLMGenerationConfig(
                    temperature=0.1,
                    max_tokens=1024,
                ),
            )

            resp = self._engine.generate(
                request=request,
                schema_class=TriageAssessment,
                max_attempts=max_attempts,
            )
            return resp.data, self._extract_telemetry(resp)

        except Exception as e:
            # Deterministic rule-based fallback ensuring enterprise resilience
            fallback = self.synthesize_deterministic_fallback(match_result, failure_reason=str(e))
            return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=str(e))

    def _build_sanitized_triage_payload(
        self,
        match_result: JobMatchResult,
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> dict[str, Any]:
        """Construct sanitized allowlist payload containing only necessary diagnostic facts."""
        job_title = "Target Position"
        company = "Target Employer"
        if jd_profile:
            job_title = scrub_pii(jd_profile.job_title or "Target Position")
            company = scrub_pii(jd_profile.company or "Target Employer")
        elif match_result.job_metadata:
            job_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Position"))
            company = scrub_pii(str(match_result.job_metadata.get("company") or "Target Employer"))

        dims = []
        for d in match_result.match_breakdown:
            dims.append({
                "dimension_name": d.dimension_name,
                "score": round(d.dimension_score, 1),
                "is_active": d.is_active,
                "status": d.evidence_status.value,
            })

        matched_skills = [scrub_pii(s.canonical_name) for s in match_result.matched_skills[:8]]
        skill_gaps = [scrub_pii(g.skill_name) for g in match_result.missing_evidence[:6]]

        return {
            "target_role": job_title,
            "target_company": company,
            "overall_score": round(match_result.overall_score, 1),
            "fit_classification": match_result.fit_classification.value,
            "active_coverage": round(match_result.active_evaluation_coverage, 1),
            "dimensions": dims,
            "verified_skills": matched_skills,
            "critical_skill_gaps": skill_gaps,
        }

    @staticmethod
    def synthesize_deterministic_fallback(
        match_result: JobMatchResult,
        failure_reason: str = "LLM unavailable",
    ) -> TriageAssessment:
        """
        Pure rule-based synthesis of TriageAssessment from deterministic scoring facts.
        """
        score = match_result.overall_score

        # 1. Determine suggested workflow goal deterministically
        if score < 60.0 or len(match_result.missing_evidence) >= 3:
            suggested_goal = UserWorkflowGoal.GAP_REMEDIATION
        elif score >= 75.0 and len(match_result.missing_evidence) <= 1:
            suggested_goal = UserWorkflowGoal.INTERVIEW_DRILL
        elif score >= 65.0:
            suggested_goal = UserWorkflowGoal.RESUME_OPTIMIZATION
        else:
            suggested_goal = UserWorkflowGoal.FULL_PREPARATION

        # 2. Extract strengths summary
        strengths = [
            f"Verified proficiency in {s.canonical_name} matching role requirements."
            for s in match_result.matched_skills[:4]
        ]
        if not strengths:
            strengths.append("Foundational professional background documented in submitted profile.")

        # 3. Extract critical gaps summary
        gaps = [
            f"Missing required competency: {g.skill_name} ({g.priority.value})."
            for g in match_result.missing_evidence[:4]
        ]
        if not gaps:
            gaps.append("No critical skill gaps identified across evaluated requirements.")

        # 4. Identify primary bottleneck
        if match_result.missing_evidence:
            top_gap = match_result.missing_evidence[0].skill_name
            bottleneck = f"Absence of verified experience with core requirement '{top_gap}' in candidate evidence."
        elif match_result.match_breakdown:
            min_dim = min(match_result.match_breakdown, key=lambda d: d.dimension_score if d.is_active else 100.0)
            bottleneck = f"Score drag in {min_dim.dimension_name} ({min_dim.dimension_score:.1f}% compatibility)."
        else:
            bottleneck = f"General alignment gap across role criteria ({score:.1f}% compatibility)."

        # 5. Recommended focus areas
        focus_areas = []
        for g in match_result.missing_evidence[:3]:
            focus_areas.append(f"Acquire hands-on project experience with {g.skill_name}.")
        if not focus_areas:
            focus_areas.append("Refine technical interview talking points around verified competencies.")

        rationale = (
            f"Deterministic evaluation yields {score:.1f}% compatibility ({match_result.fit_classification.value}). "
            f"Recommending {suggested_goal.value} to maximize interview readiness and address key bottlenecks."
        )

        return TriageAssessment(
            primary_bottleneck=bottleneck[:500],
            key_strengths_summary=tuple(strengths[:6]),
            critical_gaps_summary=tuple(gaps[:6]),
            recommended_focus_areas=tuple(focus_areas[:6]),
            suggested_goal=suggested_goal,
            advisory_rationale=rationale[:1000],
        )
