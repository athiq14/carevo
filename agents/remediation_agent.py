"""
Specialist Career Remediation Agent for Phase 5.2.
Synthesizes actionable, phased upskilling learning roadmaps and production-grade
portfolio project blueprints targeting identified employer skill gaps.
All recommended skills operate strictly in ClaimContext.ADVISORY_RECOMMENDATION.
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
from models.matching import JobDescriptionProfile, JobMatchResult, RequirementPriority
from models.resume_profile import ResumeProfile
from models.workflow import (
    CareerRemediationPlan,
    PortfolioProjectSpecification,
    UpskillingMilestone,
)

REMEDIATION_SYSTEM_PROMPT = """You are the Senior Technical Career Remediation and Upskilling Architect.
Your role is to formulate a rigorous, practical, engineering-grounded learning roadmap to bridge a candidate's skill gaps.

Guidelines:
1. Formulate 1 to 3 phased upskilling milestones (e.g. 'Days 1-30', 'Days 31-60') specifically addressing missing JD skills.
2. Formulate 1 to 2 production-grade engineering portfolio project blueprints targeting the employer's missing requirements.
3. Every milestone must have concrete, verifiable deliverables (e.g. 'Containerized service with 90% test coverage').
4. The portfolio project architecture must detail system design, data flow, tech stack, and production-grade checklists (CI/CD, tests, metrics).
5. Crucial rule: Never claim the candidate already possesses these missing skills. Frame all milestones and projects as forward-looking learning objectives.
"""


class CareerRemediationAgent(BaseSpecialistAgent):
    """
    Formulates a targeted CareerRemediationPlan addressing candidate gaps.
    Applies deterministic validation and fallback synthesis for complete reliability.
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
    ) -> tuple[CareerRemediationPlan, dict[str, Any]]:
        """
        Execute career remediation planning for candidate gaps.
        Returns (CareerRemediationPlan, telemetry_metadata).
        """
        # If user explicitly specifies Insurance Advisor, directly formulate tailored plan
        if user_career_objective and "insurance" in user_career_objective.lower():
            plan = self.synthesize_deterministic_fallback(
                match_result,
                failure_reason="Explicit user objective: Insurance Advisor",
                user_career_objective=user_career_objective,
            )
            return plan, self._extract_telemetry(None, fallback_mode=True, fallback_reason="User career objective: Insurance Advisor")

        try:
            payload = self._build_sanitized_remediation_payload(match_result, profile, jd_profile)
            user_content = (
                "Please synthesize a structured career remediation plan and project specifications for these gaps:\n\n"
                f"{self._format_safe_envelope('skill_gaps_and_role_context', payload)}"
            )

            request = LLMRequest(
                messages=[
                    LLMMessage(role=MessageRole.SYSTEM, content=REMEDIATION_SYSTEM_PROMPT),
                    LLMMessage(role=MessageRole.USER, content=user_content),
                ],
                generation_config=LLMGenerationConfig(
                    temperature=0.2,
                    max_tokens=2048,
                ),
            )

            resp = self._engine.generate(
                request=request,
                schema_class=CareerRemediationPlan,
                max_attempts=max_attempts,
            )

            plan = resp.data
            # Execute deterministic validation
            val_report = self._validator.validate_remediation_plan(plan, profile, jd_profile)
            if not val_report.is_valid:
                # Validation error in generated plan -> trigger deterministic fallback
                err_msg = "; ".join(f"[{v.violation_type.value}] {v.message}" for v in val_report.violations)
                fallback = self.synthesize_deterministic_fallback(match_result, failure_reason=err_msg, user_career_objective=user_career_objective)
                return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=err_msg)

            return plan, self._extract_telemetry(resp)

        except Exception as e:
            fallback = self.synthesize_deterministic_fallback(match_result, failure_reason=str(e), user_career_objective=user_career_objective)
            return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=str(e))

    def _build_sanitized_remediation_payload(
        self,
        match_result: JobMatchResult,
        profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> dict[str, Any]:
        """Construct sanitized allowlist payload for remediation planning."""
        job_title = "Target Position"
        if jd_profile:
            job_title = scrub_pii(jd_profile.job_title or "Target Position")
        elif match_result.job_metadata:
            job_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Position"))

        gaps = []
        for g in match_result.missing_evidence:
            gaps.append({
                "skill": scrub_pii(g.skill_name),
                "priority": g.priority.value,
                "suggested_direction": scrub_pii(g.suggested_learning_direction or "Hands-on implementation"),
            })

        verified = [scrub_pii(s.canonical_name) for s in match_result.matched_skills[:6]]

        return {
            "target_role": job_title,
            "overall_compatibility": round(match_result.overall_score, 1),
            "unverified_skill_gaps": gaps,
            "verified_candidate_competencies": verified,
        }

    @staticmethod
    def synthesize_deterministic_fallback(
        match_result: JobMatchResult,
        failure_reason: str = "LLM unavailable",
        user_career_objective: Optional[str] = None,
    ) -> CareerRemediationPlan:
        """
        Pure rule-based synthesis of CareerRemediationPlan from deterministic matching facts.
        Derives all missing requirements dynamically from JobMatchResult (including unevidenced
        partial matches such as PyTorch), separates required from preferred gaps, filters out
        already demonstrated skills, and constructs progressive milestones guaranteeing 100%
        coverage of all identified gaps without repetition.
        """
        # If user explicitly specified Insurance Advisor, construct specialized advisory blueprint
        if user_career_objective and "insurance" in user_career_objective.lower():
            m1 = UpskillingMilestone(
                timeframe="Days 1-30",
                target_missing_skills=("IRDAI Licensing (IC-38)", "Life & Health Underwriting", "Regulatory Code of Conduct"),
                learning_objectives=(
                    "Master insurance contract principles: utmost good faith, insurable interest, indemnity, and proximate cause.",
                    "Study mortality and morbidity underwriting factors, exclusion clauses, and mandatory disclosure guidelines.",
                ),
                concrete_deliverable="Attain 85%+ score on IRDAI IC-38 mock certification examinations and complete 50-hour pre-licensing training.",
            )
            m2 = UpskillingMilestone(
                timeframe="Days 31-60",
                target_missing_skills=("Client Needs Analysis", "Human Life Value Calculation", "Policy Comparison"),
                learning_objectives=(
                    "Master consultative fact-finding interview methods to evaluate household debt, income replacement, and dependent liabilities.",
                    "Build structured comparative matrices contrasting pure Term Insurance, Critical Illness Riders, and Health Cashless Policies.",
                ),
                concrete_deliverable="Deliver structured Client Fact-Finding Protocol & Product Evaluation Matrix for family risk scenarios.",
            )
            m3 = UpskillingMilestone(
                timeframe="Days 61-90",
                target_missing_skills=("Objection Handling", "Claims Servicing", "Annual Portfolio Review"),
                learning_objectives=(
                    "Develop ethical objection-handling scripts for client questions regarding pricing, claim settlement ratios, and coverage duration.",
                    "Master the end-to-end claim settlement assistance protocol, grievance redressal, and annual policy review workflows.",
                ),
                concrete_deliverable="Complete Comprehensive Family Protection Advisory Case Study & Client Onboarding Dossier.",
            )
            proj = PortfolioProjectSpecification(
                project_concept="Family Protection & Risk Advisory Portfolio: End-to-End Case Study",
                architecture_overview=(
                    "Comprehensive consultative financial risk advisory blueprint covering 4-stage client engagement: "
                    "1. Fact-finding & demographic intake, 2. Human Life Value & debt liability calculation, "
                    "3. Multi-insurer product comparison & rider optimization, 4. Claims documentation and grievance escalation workflow."
                ),
                targeted_jd_gaps=("Client Needs Analysis", "Insurance Underwriting", "Regulatory Compliance"),
                recommended_tech_stack=("IRDAI Regulatory Framework", "Human Life Value Financial Modeling", "Insurance Comparison Matrix", "CRM & Advisory Tools"),
                production_grade_checklist=(
                    "Strict compliance with IRDAI advertising and sales code of conduct.",
                    "Zero misrepresentation or unauthorized product return guarantees.",
                    "Mandatory 30-day free-look period client notification checklist.",
                    "Transparent disclosure of insurer Claim Settlement Ratios (CSR).",
                ),
            )
            return CareerRemediationPlan(
                executive_summary="Strategic career pivot plan tailored to Insurance Advisor and Financial Risk Consulting, bridging professional qualifications and consultative competencies.",
                milestones=(m1, m2, m3),
                portfolio_projects=(proj,),
                estimated_readiness_timeline="60 to 90 days to IRDAI licensing certification and active client advisory readiness.",
            )

        matched_names = {s.canonical_name.lower() for s in match_result.matched_skills}

        # 1. Separate and rank missing skills: REQUIRED first, then PREFERRED
        required_gaps: list[str] = []
        preferred_gaps: list[str] = []

        # 1a. Missing from explicit missing_evidence
        for g in match_result.missing_evidence:
            name = g.skill_name.strip()
            if name.lower() not in matched_names:
                if getattr(g, "priority", None) == RequirementPriority.REQUIRED:
                    if name not in required_gaps:
                        required_gaps.append(name)
                else:
                    if name not in preferred_gaps:
                        preferred_gaps.append(name)

        # 1b. Missing from partial_matches where direct evidence was absent (e.g. PyTorch vs TensorFlow)
        for pm in match_result.partial_matches:
            name = (pm.canonical_name or pm.skill_name).strip()
            if name.lower() not in matched_names and name not in required_gaps and name not in preferred_gaps:
                if getattr(pm, "requirement_priority", None) == RequirementPriority.REQUIRED:
                    required_gaps.append(name)
                else:
                    preferred_gaps.append(name)

        # Consolidated ordered gap list: REQUIRED first, then PREFERRED
        ordered_gaps = required_gaps + [p for p in preferred_gaps if p not in required_gaps]
        if not ordered_gaps:
            ordered_gaps = ["Distributed Systems Architecture", "Production Cloud Infrastructure"]

        # 2. Extract verified skills to leverage as engineering foundations
        candidate_known = [s.canonical_name for s in match_result.matched_skills[:3]] or ["Python"]
        role_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Role"))

        # 3. Dynamically partition ALL missing gaps across sequential milestones (up to 5 milestones)
        # Guarantees 100% gap coverage, zero omission, and zero duplication across milestones
        total_gaps = len(ordered_gaps)
        import math
        if total_gaps <= 1:
            # Single gap requires 2 progressive stages: Core Fundamentals -> Production Integration
            chunks = [[ordered_gaps[0]], [ordered_gaps[0]]]
        elif total_gaps <= 4:
            chunk_size = math.ceil(total_gaps / 2)
            chunks = [ordered_gaps[i:i + chunk_size] for i in range(0, total_gaps, chunk_size)]
        elif total_gaps <= 7:
            chunk_size = math.ceil(total_gaps / 3)
            chunks = [ordered_gaps[i:i + chunk_size] for i in range(0, total_gaps, chunk_size)]
        elif total_gaps <= 12:
            chunk_size = math.ceil(total_gaps / 4)
            chunks = [ordered_gaps[i:i + chunk_size] for i in range(0, total_gaps, chunk_size)]
        else:
            num_milestones = min(5, total_gaps)
            chunk_size = math.ceil(total_gaps / num_milestones)
            chunks = [ordered_gaps[i:i + chunk_size] for i in range(0, total_gaps, chunk_size)]

        milestones: list[UpskillingMilestone] = []
        timeframe_labels = ["Days 1-30", "Days 31-60", "Days 61-90", "Days 91-120", "Days 121-150"]

        for m_idx, m_targets in enumerate(chunks):
            if not m_targets:
                continue

            tf_label = timeframe_labels[m_idx] if m_idx < len(timeframe_labels) else f"Phase {m_idx + 1}"
            lead_skill = m_targets[0]
            skills_joined = ", ".join(m_targets)

            if total_gaps == 1 and m_idx == 1:
                objs = [
                    f"Implement advanced production integration, reliability, and deployment pipelines for {lead_skill}.",
                    f"Design end-to-end monitoring, configuration management, and failover workflows for {lead_skill}.",
                    f"Progressively integrate {lead_skill} into production-grade infrastructure.",
                ]
                deliverable = f"Production repository deliverable demonstrating {lead_skill} cluster deployment and automated integration test suites."
            else:
                objs = [
                    f"Master architectural design, syntax idioms, and performance patterns for {skills_joined}.",
                    f"Implement isolated components and integration test suites validating {lead_skill}.",
                ]
                if m_idx == 0:
                    objs.append(f"Leverage candidate's verified background in {candidate_known[0]} to accelerate adoption of {lead_skill}.")
                else:
                    prev_lead = milestones[-1].target_missing_skills[0]
                    objs.append(f"Progressively integrate {skills_joined} into services established with {prev_lead}.")

                deliverable = f"Production repository deliverable demonstrating {skills_joined} with unit and integration test suites."

            m = UpskillingMilestone(
                timeframe=tf_label,
                target_missing_skills=tuple(m_targets),
                learning_objectives=tuple(objs),
                concrete_deliverable=deliverable[:500],
            )
            milestones.append(m)

        # 4. Portfolio Project Specification targeting primary missing gaps
        targeted_project_gaps = ordered_gaps[:min(len(ordered_gaps), 10)]
        project_stack = (candidate_known + ordered_gaps[:6])[:12]

        proj = PortfolioProjectSpecification(
            project_concept=f"Production {targeted_project_gaps[0]} & {targeted_project_gaps[1] if len(targeted_project_gaps) > 1 else 'Systems'} Engineering Blueprint",
            architecture_overview=(
                f"Distributed asynchronous service combining candidate's background in {', '.join(candidate_known)} "
                f"with newly acquired proficiency in {', '.join(targeted_project_gaps[:4])}. Architecture includes decoupled worker nodes, "
                "event streaming, containerized orchestration, structured logging, error recovery, and automated test fixtures."
            ),
            targeted_jd_gaps=tuple(targeted_project_gaps),
            recommended_tech_stack=tuple(project_stack),
            production_grade_checklist=tuple([
                "Unit and integration test coverage exceeding 85%",
                "Containerized deployment via Dockerfile and Kubernetes manifests",
                "Automated linting, formatting, and CI/CD pipeline checks",
                "Benchmark validation demonstrating low-latency throughput metrics",
            ]),
        )

        timeline = f"{len(milestones) * 4}-{len(milestones) * 4 + 4} weeks (Estimated effort: 12-15 hrs/week self-paced; not a guaranteed hiring outcome)"
        summary = (
            f"Strategic upskilling trajectory designed to bridge all {len(ordered_gaps)} identified requirement gap(s) "
            f"({', '.join(ordered_gaps[:5])}{'...' if len(ordered_gaps) > 5 else ''}) for {role_title}. "
            "Prioritizes verifiable portfolio evidence and hands-on deliverables over passive study."
        )

        return CareerRemediationPlan(
            executive_summary=summary[:1000],
            milestones=tuple(milestones),
            portfolio_projects=(proj,),
            estimated_readiness_timeline=timeline[:200],
        )
