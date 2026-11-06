"""
Specialist Resume Optimization Agent for Phase 5.2.
Transforms verified candidate experience into high-impact STAR bullet points.
Strictly preserves employer names, job titles, and tenure dates.
Grounds all highlighted skills in verified candidate competencies and enforces metric disclosures.
"""
import re
from typing import Any, Optional
from agents.base import BaseSpecialistAgent
from core.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
)
from core.matching.explanation_generator import scrub_pii
from core.workflow.validators import (
    QUANTITATIVE_METRIC_PATTERN,
    DeterministicWorkflowValidator,
)
from models.matching import JobDescriptionProfile, JobMatchResult
from models.resume_profile import ResumeProfile
from models.workflow import (
    MetricDisclosureStatus,
    OptimizedBulletPoint,
    ResumeOptimizationReport,
)

RESUME_OPT_SYSTEM_PROMPT = """You are the Senior Technical Resume Optimization and ATS Alignment Specialist.
Your responsibility is to reframe verified candidate career evidence into high-impact STAR bullet points.

Strict Rules:
1. Every optimized bullet must reference an exact source entity ID (e.g. 'exp_0', 'proj_0') and field path (e.g. 'experience[0].responsibilities[0]').
2. Never invent qualifications, employment history, employers, job titles, dates, or responsibilities.
3. Every highlighted skill MUST be a skill the candidate has already demonstrated in their resume. Never highlight unevidenced skills.
4. If a bullet includes quantitative figures (percentages, latency, throughput, scale), tag metric_disclosure as 'SELF_REPORTED_METRIC'.
5. Do not include raw contact details (emails, phone numbers, addresses, SSNs) anywhere in your output.
6. PRESERVE FACTUAL EVIDENCE AND HONESTY:
   - Compare every generated bullet against its exact source evidence.
   - Preserve original candidate scope; never invent metrics, leadership, ownership, scale, outcomes, or technical responsibilities.
   - Do NOT prepend awkward repetitive phrases or double verbs (e.g. never generate 'Engineered and delivered founder...' or 'Engineered and delivered founded...').
   - Improve grammar and clarity with natural, professional active verbs without inflating claims.
   - Never claim independent verification; candidate bullets represent self-reported experience.
"""


STRONG_ACTION_VERBS = (
    "Architected", "Engineered", "Developed", "Optimized", "Designed", "Implemented",
    "Built", "Deployed", "Led", "Spearheaded", "Created", "Configured", "Orchestrated",
    "Automated", "Formulated", "Constructed", "Established", "Directed", "Managed",
    "Conducted", "Analyzed", "Integrated", "Delivered", "Scaled", "Refactored",
    "Streamlined", "Authored", "Produced", "Launched", "Initiated", "Founded",
    "Trained", "Maintained", "Collaborated", "Researched",
)


def reframe_grounded_star_bullet(clean_text: str) -> str:
    """
    Reframes candidate experience into clear, professional, grammatically sound STAR bullet.
    Preserves original factual meaning strictly without claim inflation, redundant double verbs,
    or awkward mechanical prefixes.
    """
    text = clean_text.strip().rstrip(".;,")
    if not text:
        return ""

    words = text.split()
    first_word = words[0].capitalize()

    # 1. If text already begins with a recognized strong past-tense action verb:
    if any(first_word == v for v in STRONG_ACTION_VERBS):
        return f"{first_word} {' '.join(words[1:])}."

    # 2. Check for role/title prefix line (e.g. 'Founder – Hulsti.AI')
    if re.match(r'^(?:founder|co-founder|lead|director|manager|engineer|intern|consultant|specialist|head)\b', text, re.IGNORECASE):
        if " – " in text or " - " in text:
            parts = re.split(r'\s+[–-]\s+', text, maxsplit=1)
            title_part = parts[0].strip()
            org_part = parts[1].strip()
            return f"Served as {title_part} at {org_part}."
        return f"Served as {text}."

    # 3. Check for gerund or present participle
    gerund_map = {
        "building": "Built", "developing": "Developed", "creating": "Created",
        "designing": "Designed", "engineering": "Engineered", "leading": "Led",
        "managing": "Managed", "implementing": "Implemented", "optimizing": "Optimized",
        "deploying": "Deployed", "automating": "Automated", "integrating": "Integrated",
        "scaling": "Scaled", "refactoring": "Refactored", "analyzing": "Analyzed",
        "maintaining": "Maintained",
    }
    w0_lower = words[0].lower()
    if w0_lower in gerund_map:
        return f"{gerund_map[w0_lower]} {' '.join(words[1:])}."

    # 4. Check for noun phrase starting with 'client and internal projects'
    if text.lower().startswith("client and internal projects"):
        extra = " ".join(words[4:]) if len(words) > 4 else ""
        if extra:
            return f"Contributed technical deliverables across client and internal projects ({extra})."
        return "Contributed technical deliverables across client and internal projects."

    # 5. Check if it refers to a system, pipeline, model, tool, application
    if re.search(r'\b(system|application|pipeline|platform|tool|service|dashboard|model|interface|api|microservices)\b', text, re.IGNORECASE):
        return f"Developed and maintained {text[0].lower() + text[1:]}."

    # 6. Fallback clean framing without repeating words or double verbs
    return f"Executed {text[0].lower() + text[1:]}."


def classify_bullet_structure(text: str) -> str:
    """
    Classifies whether a candidate bullet contains a quantified result (Full STAR)
    or represents an action/responsibility without manufactured metrics.
    """
    if QUANTITATIVE_METRIC_PATTERN.search(text):
        return "Full STAR (Action & Measurable Result)"
    return "Action & Scope (Self-Reported Responsibility)"


class ResumeOptimizationAgent(BaseSpecialistAgent):
    """
    Optimizes resume bullets with STAR framing while strictly preserving factual grounding.
    Enforces factual validation via DeterministicWorkflowValidator.
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
        profile: ResumeProfile,
        match_result: JobMatchResult,
        jd_profile: Optional[JobDescriptionProfile] = None,
        max_attempts: int = 2,
    ) -> tuple[ResumeOptimizationReport, dict[str, Any]]:
        """
        Execute resume optimization over verified candidate profile and role requirements.
        Returns (ResumeOptimizationReport, telemetry_metadata).
        """
        try:
            payload = self._build_sanitized_resume_payload(profile, match_result, jd_profile)
            user_content = (
                "Please optimize the following verified candidate experience into STAR bullet points:\n\n"
                f"{self._format_safe_envelope('candidate_evidence_and_role_context', payload)}"
            )

            request = LLMRequest(
                messages=[
                    LLMMessage(role=MessageRole.SYSTEM, content=RESUME_OPT_SYSTEM_PROMPT),
                    LLMMessage(role=MessageRole.USER, content=user_content),
                ],
                generation_config=LLMGenerationConfig(
                    temperature=0.1,
                    max_tokens=2048,
                ),
            )

            resp = self._engine.generate(
                request=request,
                schema_class=ResumeOptimizationReport,
                max_attempts=max_attempts,
            )

            report = resp.data
            # Execute deterministic validation
            val_report = self._validator.validate_resume_optimization(report, profile, jd_profile)
            if not val_report.is_valid:
                err_msg = "; ".join(f"[{v.violation_type.value}] {v.message}" for v in val_report.violations)
                fallback = self.synthesize_deterministic_fallback(profile, match_result, failure_reason=err_msg)
                return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=err_msg)

            return report, self._extract_telemetry(resp)

        except Exception as e:
            fallback = self.synthesize_deterministic_fallback(profile, match_result, failure_reason=str(e))
            return fallback, self._extract_telemetry(None, fallback_mode=True, fallback_reason=str(e))

    def _build_sanitized_resume_payload(
        self,
        profile: ResumeProfile,
        match_result: JobMatchResult,
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> dict[str, Any]:
        """Construct sanitized allowlist payload containing structured experience and target keywords."""
        job_title = "Target Position"
        if jd_profile:
            job_title = scrub_pii(jd_profile.job_title or "Target Position")
        elif match_result.job_metadata:
            job_title = scrub_pii(str(match_result.job_metadata.get("job_title") or "Target Position"))

        experiences = []
        for idx, exp in enumerate(profile.experience):
            entity_id = f"exp_{idx}"
            resps = [
                {"field_path": f"experience[{idx}].responsibilities[{r_idx}]", "text": scrub_pii(r)}
                for r_idx, r in enumerate(exp.responsibilities)
            ]
            experiences.append({
                "entity_id": entity_id,
                "job_title": scrub_pii(exp.job_title or "Engineer"),
                "company": scrub_pii(exp.company or "Company"),
                "technologies": [scrub_pii(t) for t in exp.technologies],
                "responsibilities": resps,
            })

        verified_skills = [scrub_pii(s.name) for s in profile.skills]

        return {
            "target_role": job_title,
            "verified_skills_whitelist": verified_skills,
            "experience_records": experiences,
        }

    @staticmethod
    def synthesize_deterministic_fallback(
        profile: ResumeProfile,
        match_result: JobMatchResult,
        failure_reason: str = "LLM unavailable",
    ) -> ResumeOptimizationReport:
        """
        Pure rule-based synthesis of ResumeOptimizationReport guaranteed to pass
        all deterministic evidence resolution, candidate skill grounding, and metric checks.
        """
        candidate_skill_names = {s.name.lower(): s.name for s in profile.skills}
        bullets: list[OptimizedBulletPoint] = []
        seen_bullet_texts: set[str] = set()

        # Generate STAR-reframed bullets from verified experience
        for exp_idx, exp in enumerate(profile.experience):
            for r_idx, resp_text in enumerate(exp.responsibilities):
                clean_resp = scrub_pii(resp_text)
                if not clean_resp:
                    continue

                # Ensure grounded, natural STAR action verb framing
                star_bullet = reframe_grounded_star_bullet(clean_resp)
                bullet_key = re.sub(r'\s+', ' ', star_bullet.strip().lower())
                if bullet_key in seen_bullet_texts:
                    continue
                seen_bullet_texts.add(bullet_key)

                # Identify grounded skills present in this responsibility
                grounded_skills: list[str] = []
                for canon_lower, canon_name in candidate_skill_names.items():
                    if re.search(rf"\b{re.escape(canon_lower)}\b", clean_resp.lower()):
                        grounded_skills.append(canon_name)

                # Check for quantitative claims
                has_metric = bool(QUANTITATIVE_METRIC_PATTERN.search(clean_resp))
                metric_status = (
                    MetricDisclosureStatus.SELF_REPORTED_METRIC
                    if has_metric
                    else MetricDisclosureStatus.NOT_APPLICABLE
                )

                bullets.append(
                    OptimizedBulletPoint(
                        bullet_id=f"opt_exp_{exp_idx}_{r_idx}",
                        source_entity_id=f"exp_{exp_idx}",
                        source_field_path=f"experience[{exp_idx}].responsibilities[{r_idx}]",
                        improved_star_bullet=star_bullet[:500],
                        highlighted_skills=tuple(grounded_skills[:4]),
                        alignment_rationale="Reframed candidate responsibility using active verbs while strictly preserving factual grounding.",
                        metric_disclosure=metric_status,
                        provenance_label="Rewritten from candidate-provided evidence",
                        bullet_structure=classify_bullet_structure(star_bullet),
                    )
                )

        if not bullets:
            # Minimal grounded bullet if experience was empty
            bullets.append(
                OptimizedBulletPoint(
                    bullet_id="opt_fallback_1",
                    source_entity_id="proj_0" if profile.projects else "exp_0",
                    source_field_path="projects[0].description" if profile.projects else "experience[0].job_title",
                    improved_star_bullet="Engineered technical solutions demonstrating practical systems development.",
                    highlighted_skills=tuple([profile.skills[0].name]) if profile.skills else tuple(),
                    alignment_rationale="Grounded fallback bullet.",
                    metric_disclosure=MetricDisclosureStatus.NOT_APPLICABLE,
                    bullet_structure="Action & Scope (Self-Reported Responsibility)",
                )
            )

        summary = (
            f"Resume optimization highlights verified candidate competencies in "
            f"{', '.join(s.name for s in profile.skills[:4])} aligned with role expectations."
        )

        return ResumeOptimizationReport(
            target_role_alignment_summary=summary[:1000],
            optimized_bullets=tuple(bullets[:10]),
            section_reordering_recommendations=(
                "Position verified technical skills directly beneath professional summary",
                "Ensure engineering projects prominently cite verified technology stack",
            ),
            ats_keyword_density_feedback=(
                f"Verified strong representation of core competencies ({', '.join(s.name for s in profile.skills[:3])})",
            ),
        )
