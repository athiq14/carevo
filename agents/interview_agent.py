"""
Specialist Interview Preparation Agent for Phase 5.2.
Synthesizes role-specific interview preparation kits including technical deep dives,
gap mitigation inquiries, metric challenge probes, and candidate questions.
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
from core.workflow.validators import DeterministicWorkflowValidator
from models.matching import JobDescriptionProfile, JobMatchResult
from models.resume_profile import ResumeProfile
from models.workflow import (
    InterviewPreparationKit,
    InterviewQuestion,
    InterviewQuestionCategory,
)

INTERVIEW_PREP_SYSTEM_PROMPT = """You are the Senior Technical Interview Coach and Assessment Architect.
Your responsibility is to formulate a role-specific, evidence-grounded interview preparation kit.

Guidelines:
1. Formulate technical deep-dive questions targeting the candidate's verified competencies.
2. Formulate gap-mitigation questions targeting unevidenced or missing JD requirements, coaching the candidate on addressing adjacencies without misrepresenting capabilities.
3. Formulate metric-challenge questions probing self-reported quantitative accomplishments.
4. Include grounded talking points for each question connecting back to candidate experience.
5. Provide thoughtful questions for the candidate to ask the interviewer.
6. Crucial rule: Do not claim the candidate possesses unevidenced skills. Do not include raw personal contact details.
"""


class InterviewPreparationAgent(BaseSpecialistAgent):
    """
    Formulates a tailored InterviewPreparationKit using verified candidate evidence and role demands.
    Enforces deterministic validation and provides robust rule-based fallback.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        dispatcher: Optional[Any] = None,
    ) -> None:
        super().__init__(engine=engine, dispatcher=dispatcher)
        self._validator = DeterministicWorkflowValidator()

    def run(
        self,
        match_result: JobMatchResult,
        profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
        max_attempts: int = 2,
        user_career_objective: Optional[str] = None,
    ) -> tuple[InterviewPreparationKit, dict[str, Any]]:
        """
        Execute interview preparation kit formulation.
        Returns (InterviewPreparationKit, telemetry_metadata).
        """
        # If user explicitly specifies Insurance Advisor, directly formulate tailored kit
        if user_career_objective and "insurance" in user_career_objective.lower():
            kit = self.synthesize_deterministic_fallback(
                match_result,
                profile,
                failure_reason="Explicit user objective: Insurance Advisor",
                user_career_objective=user_career_objective,
            )
            return kit, self._extract_telemetry(None, fallback_mode=True, fallback_reason="User career objective: Insurance Advisor")

        try:
            payload = self._build_sanitized_interview_payload(match_result, profile, jd_profile)
            user_content = (
                "Please formulate a structured interview preparation kit based on these matching facts:\n\n"
                f"{self._format_safe_envelope('interview_evidence_and_gap_context', payload)}"
            )

            request = LLMRequest(
                messages=[
                    LLMMessage(role=MessageRole.SYSTEM, content=INTERVIEW_PREP_SYSTEM_PROMPT),
                    LLMMessage(role=MessageRole.USER, content=user_content),
                ],
                generation_config=LLMGenerationConfig(
                    temperature=0.2,
                    max_tokens=2048,
                ),
            )

            resp = self._engine.generate(
                request=request,
                schema_class=InterviewPreparationKit,
                max_attempts=max_attempts,
            )

            kit = resp.data
            # Execute deterministic validation
            val_report = self._validator.validate_interview_kit(kit, profile, jd_profile)
            if not val_report.is_valid:
                err_msg = "; ".join(f"[{v.violation_type.value}] {v.message}" for v in val_report.violations)
                fallback = self.synthesize_deterministic_fallback(match_result, profile, failure_reason=err_msg, user_career_objective=user_career_objective)
                return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=err_msg)

            return kit, self._extract_telemetry(resp)

        except Exception as e:
            fallback = self.synthesize_deterministic_fallback(match_result, profile, failure_reason=str(e), user_career_objective=user_career_objective)
            return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=str(e))

    def _build_sanitized_interview_payload(
        self,
        match_result: JobMatchResult,
        profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> dict[str, Any]:
        """Construct sanitized allowlist payload for interview preparation."""
        job_title = "Target Position"
        if jd_profile:
            job_title = scrub_pii(jd_profile.job_title or "Target Position")
        elif match_result.job_metadata:
            job_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Position"))

        verified_skills = [scrub_pii(s.canonical_name) for s in match_result.matched_skills[:6]]
        gaps = [scrub_pii(g.skill_name) for g in match_result.missing_evidence[:4]]

        projects = []
        for p in match_result.project_evidence[:2]:
            projects.append({
                "project_name": scrub_pii(p.project_name),
                "technologies": [scrub_pii(t) for t in p.relevant_technologies],
                "self_reported_metrics": [scrub_pii(m) for m in p.self_reported_metrics[:2]],
            })

        return {
            "target_role": job_title,
            "verified_competencies": verified_skills,
            "missing_requirements": gaps,
            "project_evidence": projects,
        }

    @staticmethod
    def synthesize_deterministic_fallback(
        match_result: JobMatchResult,
        profile: ResumeProfile,
        failure_reason: str = "LLM unavailable",
        user_career_objective: Optional[str] = None,
    ) -> InterviewPreparationKit:
        """
        Pure rule-based synthesis of InterviewPreparationKit guaranteed to pass
        deterministic evidence resolution and PII validation.
        """
        if user_career_objective and "insurance" in user_career_objective.lower():
            tech_q = InterviewQuestion(
                question_id="tech_q_1",
                category=InterviewQuestionCategory.TECHNICAL,
                question_text="When conducting a client fact-finding interview, how do you calculate Human Life Value (HLV) and evaluate life vs. health coverage requirements?",
                interviewer_intent="Assess structured consultative inquiry, quantitative risk calculation, and customer empathy in Insurance Advisory.",
                targeted_skill_or_gap="Human Life Value Calculation",
                suggested_talking_points=(
                    "Explain the Human Life Value calculation factoring annual income, remaining working years, and debt liabilities.",
                    "Discuss balancing term life protection for pure mortality risk with comprehensive health insurance for critical illnesses.",
                    "Emphasize ethical disclosure and avoiding product mis-selling.",
                ),
                grounded_source_entity_id="proj_0" if profile.projects else ("exp_0" if profile.experience else None),
            )
            gap_q = InterviewQuestion(
                question_id="gap_q_1",
                category=InterviewQuestionCategory.GAP_PROBE,
                question_text="As you transition into the Insurance Advisor profession, what is your approach to mastering IRDAI regulations (IC-38) and compliance disclosures?",
                interviewer_intent="Evaluate compliance awareness, ethical commitment, and learning discipline under financial regulatory frameworks.",
                targeted_skill_or_gap="IRDAI Regulatory Compliance",
                suggested_talking_points=(
                    "Acknowledge active preparation for the IRDAI IC-38 licensing examination with dedicated study of code of conduct and sales ethics.",
                    "Draw upon analytical foundation from computer science to manage structured policy comparison and documentation accuracy.",
                    "Detail adherence to 30-day free-look provisions, mandatory disclosures, and transparent claim settlement ratios.",
                ),
                grounded_source_entity_id=None,
            )
            metric_q = InterviewQuestion(
                question_id="metric_q_1",
                category=InterviewQuestionCategory.METRIC_CHALLENGE,
                question_text="In consultative insurance advisory, how do you measure client satisfaction, persistency ratio, and claim assistance turnaround?",
                interviewer_intent="Probe understanding of long-term advisory metrics, client retention, and post-sale servicing.",
                targeted_skill_or_gap="Persistency & Client Advisory Metrics",
                suggested_talking_points=(
                    "Emphasize the importance of 13th and 25th month persistency ratios as indicators of honest, needs-based sales.",
                    "Discuss establishing regular annual policy check-ins to update beneficiary designations and lifestyle changes.",
                    "Highlight supporting families proactively during claims processing.",
                ),
                grounded_source_entity_id="proj_0" if profile.projects else None,
            )
            overview = "Consultative interview preparation kit tailored for Insurance Advisor and Financial Risk Specialist positions, focusing on client needs analysis, IRDAI compliance, and ethical advisory."
            interviewer_questions = (
                "What continuing education and advanced product training (commercial lines, critical illness) does the firm provide to newly licensed advisors?",
                "How does the branch operations team assist advisors in facilitating cashless hospital claims for policyholders?",
                "What digital tools or CRM platforms are provided to streamline client needs discovery and policy comparison?",
            )
            return InterviewPreparationKit(
                role_context_overview=overview[:1000],
                technical_deep_dive_questions=(tech_q,),
                gap_mitigation_questions=(gap_q,),
                metric_challenge_questions=(metric_q,),
                recommended_questions_for_interviewer=interviewer_questions,
            )

        top_skills = [s.canonical_name for s in match_result.matched_skills[:3]] or ["Software Engineering"]
        primary_skill = top_skills[0]
        role_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Position"))

        # 1. Technical Deep Dive Question (Grounded in actual candidate project or verified experience)
        if profile.projects:
            p0 = profile.projects[0]
            proj_techs = ", ".join(p0.technologies[:3]) or primary_skill
            tech_q = InterviewQuestion(
                question_id="tech_q_1",
                category=InterviewQuestionCategory.TECHNICAL,
                question_text=(
                    f"In your engineering project '{p0.project_name}', how did you architect the solution using {proj_techs}, "
                    "and what technical trade-offs did you make regarding reliability and modularity?"
                ),
                interviewer_intent=f"Assess hands-on architectural decision-making, component boundaries, and testing in '{p0.project_name}'.",
                targeted_skill_or_gap=p0.technologies[0] if p0.technologies else primary_skill,
                suggested_talking_points=(
                    f"Walk through the architecture and design decisions of '{p0.project_name}' without inflating claims.",
                    f"Discuss trade-offs made when implementing {proj_techs} and how errors and exceptions were handled.",
                    "Highlight unit and integration test coverage implemented in the codebase.",
                ),
                grounded_source_entity_id="proj_0",
            )
        else:
            tech_q = InterviewQuestion(
                question_id="tech_q_1",
                category=InterviewQuestionCategory.TECHNICAL,
                question_text=f"Can you explain the concurrency or architectural patterns you employ when developing services with {primary_skill}?",
                interviewer_intent=f"Assess depth of hands-on experience and architectural understanding of {primary_skill}.",
                targeted_skill_or_gap=primary_skill,
                suggested_talking_points=(
                    f"Highlight verified hands-on projects and responsibilities where {primary_skill} was used.",
                    "Discuss trade-offs made between throughput, maintainability, and code complexity.",
                ),
                grounded_source_entity_id="exp_0" if profile.experience else None,
            )

        # 2. Gap Mitigation Question (Honest probe on missing requirements without misrepresenting experience)
        missing_skills = [g.skill_name for g in match_result.missing_evidence[:2]]
        gap_target = missing_skills[0] if missing_skills else "Distributed Cloud Infrastructure"
        gap_q = InterviewQuestion(
            question_id="gap_q_1",
            category=InterviewQuestionCategory.GAP_PROBE,
            question_text=(
                f"This {role_title} role emphasizes {gap_target}, which is currently an upskilling focus for you. "
                f"Given your verified background in {primary_skill}, what is your technical strategy for ramping up "
                f"on {gap_target} quickly, and what architectural analogies do you draw between them?"
            ),
            interviewer_intent=f"Evaluate learning agility, technical adaptability, and conceptual transfer to {gap_target}.",
            targeted_skill_or_gap=gap_target,
            suggested_talking_points=(
                f"Transparently acknowledge {gap_target} as an active upskilling priority without claiming unevidenced production tenure.",
                f"Draw direct analogies from proven engineering principles in {primary_skill} to demonstrate conceptual readiness.",
                "Detail your structured learning plan (hands-on prototyping, test-driven implementations) to bridge the gap rapidly.",
            ),
            grounded_source_entity_id=None,
        )

        # 3. Metric Challenge Question
        metric_q = InterviewQuestion(
            question_id="metric_q_1",
            category=InterviewQuestionCategory.METRIC_CHALLENGE,
            question_text="How did you measure, validate, and benchmark the performance and reliability of your software implementations?",
            interviewer_intent="Probe whether quantitative claims and outcomes are grounded in verifiable benchmarks and tests.",
            targeted_skill_or_gap="Performance Engineering",
            suggested_talking_points=(
                "Clarify that metrics reflect self-reported internal testing and operational telemetry.",
                "Walk through the baseline measurement and post-optimization delta.",
            ),
            grounded_source_entity_id="proj_0" if profile.projects else None,
        )

        overview = (
            f"Technical interview preparation targeting {primary_skill} depth, "
            f"objective gap mitigation for {gap_target}, and structured metric articulation."
        )

        interviewer_questions = (
            "What architectural challenges is the engineering team currently solving?",
            "What does the testing and deployment pipeline look like for this service?",
            "How does the team prioritize technical debt versus new feature development?",
        )

        return InterviewPreparationKit(
            role_context_overview=overview[:1000],
            technical_deep_dive_questions=(tech_q,),
            gap_mitigation_questions=(gap_q,),
            metric_challenge_questions=(metric_q,),
            recommended_questions_for_interviewer=interviewer_questions,
        )
