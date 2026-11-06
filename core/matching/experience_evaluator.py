"""
Deterministic experience evaluator for candidate-job matching.
Separates Minimum Employment Tenure from Role/Domain Relevance.
Implements approved discrete experience tiers, entry-level special rubrics,
and precedence resolution for mixed candidate employment backgrounds.
"""
from datetime import datetime, timezone
import re
from typing import Optional

from models.matching import (
    EvidenceStatus,
    ExperienceTier,
    JobDescriptionProfile,
)
from models.resume_profile import ResumeProfile
from models.resume_entities import WorkExperience


# Standard Experience Tier Scoring Rubric
TIER_SCORES: dict[ExperienceTier, float] = {
    ExperienceTier.DIRECT_ALIGNMENT: 100.0,
    ExperienceTier.ADJACENT_ALIGNMENT: 75.0,
    ExperienceTier.RELEVANT_INTERNSHIP: 60.0,       # 80.0% if JD is entry-level
    ExperienceTier.APPLIED_PROJECTS: 40.0,
    ExperienceTier.UNRELATED_BACKGROUND: 0.0,
    ExperienceTier.UNKNOWN: 0.0,                   # Sub-component neutralized
}


class ExperienceEvaluator:
    """
    Evaluator responsible for assessing candidate tenure and technical role relevance.
    Applies strict precedence rules and prevents non-technical tenure from inflating relevance scores.
    """

    def evaluate(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
    ) -> dict[str, object]:
        """
        Evaluate candidate work experience against job description requirements.

        Returns:
            dict containing:
                - dimension_score: float bounded [0.0, 100.0]
                - tenure_score: Optional[float]
                - relevance_score: Optional[float]
                - assigned_tier: ExperienceTier
                - candidate_tenure_years: float
                - required_tenure_years: Optional[float]
                - is_active: bool
                - neutralization_explanation: Optional[str]
                - evidence_status: EvidenceStatus
                - diagnostics: list[str]
        """
        diagnostics: list[str] = []

        # Step 1: Parse Required Minimum Tenure from JD
        required_years = self._parse_required_tenure(jd_profile.experience_requirement)

        # Step 2: Calculate Candidate Technical Full-Time Employment Tenure
        candidate_ft_years, relevant_jobs, unrelated_jobs, internships = self._analyze_career_history(
            resume_profile, jd_profile
        )

        # Step 3: Evaluate Tenure Sub-component (S_tenure)
        tenure_score: Optional[float] = None
        tenure_active = (required_years is not None and required_years > 0.0)

        if tenure_active:
            if candidate_ft_years >= required_years:
                tenure_score = 100.0
                diagnostics.append(
                    f"Tenure requirement satisfied: {candidate_ft_years:.1f} verified full-time years >= {required_years:.1f} years required."
                )
            else:
                tenure_score = (candidate_ft_years / required_years) * 100.0
                diagnostics.append(
                    f"Partial tenure credit: {candidate_ft_years:.1f} verified full-time years / {required_years:.1f} years required ({tenure_score:.1f}%)."
                )
        else:
            diagnostics.append("Minimum professional experience tenure not specified by employer; tenure sub-weight neutralized.")

        # Step 4: Evaluate Role Relevance Sub-component (S_relevance)
        # Check if JD specifies a target role or core responsibilities
        has_jd_target = bool(jd_profile.job_title and jd_profile.job_title.strip()) or len(jd_profile.responsibilities) > 0
        assigned_tier: ExperienceTier
        relevance_score: Optional[float] = None
        relevance_active = True

        if not has_jd_target:
            # Scenario 3: JD omits target role and core duties -> Role Relevance is UNKNOWN_EVIDENCE
            assigned_tier = ExperienceTier.UNKNOWN
            relevance_active = False
            diagnostics.append(
                "Job description omits target role title and duties; role relevance cannot be assessed and is neutralized."
            )
        else:
            # Determine Tier via Precedence Matrix
            assigned_tier = self._determine_precedence_tier(
                resume_profile=resume_profile,
                jd_profile=jd_profile,
                relevant_jobs=relevant_jobs,
                unrelated_jobs=unrelated_jobs,
                internships=internships,
                required_years=required_years,
            )

            # Apply entry-level special rule for Tier 3
            is_entry_level = (required_years == 0.0 or required_years is None or self._is_entry_level_jd(jd_profile))
            if assigned_tier == ExperienceTier.RELEVANT_INTERNSHIP and is_entry_level:
                relevance_score = 80.0
                diagnostics.append("Relevant internship evaluated under Entry-Level JD rubric: awarded 80.0% relevance.")
            elif assigned_tier == ExperienceTier.UNKNOWN:
                relevance_active = False
                relevance_score = None
            else:
                relevance_score = TIER_SCORES[assigned_tier]
                diagnostics.append(f"Role relevance assigned {assigned_tier.value}: {relevance_score:.1f}%.")

        # Step 5: Sub-component Combination and Neutralization Redistribution
        if not tenure_active and not relevance_active:
            # Both unstated/unassessable -> Experience dimension is completely neutralized
            return {
                "dimension_score": 0.0,
                "tenure_score": None,
                "relevance_score": None,
                "assigned_tier": ExperienceTier.UNKNOWN,
                "candidate_tenure_years": candidate_ft_years,
                "required_tenure_years": required_years,
                "is_active": False,
                "neutralization_explanation": "Job description omits both minimum experience tenure and target role duties.",
                "evidence_status": EvidenceStatus.UNKNOWN_EVIDENCE,
                "diagnostics": diagnostics,
            }

        dimension_score: float
        if tenure_active and relevance_active:
            # Both active: 50% tenure, 50% relevance
            assert tenure_score is not None and relevance_score is not None
            dimension_score = 0.50 * tenure_score + 0.50 * relevance_score
            ev_status = EvidenceStatus.VERIFIED_EVIDENCE if dimension_score >= 75.0 else EvidenceStatus.PARTIAL_EVIDENCE
        elif tenure_active and not relevance_active:
            # Relevance neutralized: 100% on tenure (Scenario 3)
            assert tenure_score is not None
            dimension_score = tenure_score
            ev_status = EvidenceStatus.VERIFIED_EVIDENCE if tenure_score >= 100.0 else EvidenceStatus.PARTIAL_EVIDENCE
        else:
            # Tenure neutralized: 100% on relevance (Scenario 4)
            assert relevance_score is not None
            dimension_score = relevance_score
            ev_status = EvidenceStatus.VERIFIED_EVIDENCE if relevance_score >= 75.0 else EvidenceStatus.PARTIAL_EVIDENCE

        dimension_score = max(0.0, min(100.0, dimension_score))

        return {
            "dimension_score": dimension_score,
            "tenure_score": tenure_score,
            "relevance_score": relevance_score,
            "assigned_tier": assigned_tier,
            "candidate_tenure_years": candidate_ft_years,
            "required_tenure_years": required_years,
            "is_active": True,
            "neutralization_explanation": None,
            "evidence_status": ev_status,
            "diagnostics": diagnostics,
        }

    def _parse_required_tenure(self, exp_req_text: Optional[str]) -> Optional[float]:
        """Parse numerical minimum years of required experience from JD requirement text."""
        if not exp_req_text:
            return None
        text = exp_req_text.lower()
        # Look for explicit patterns like "3+ years", "3 to 5 years", "3 yrs"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+|to\s*\d+)?\s*(?:years?|yrs?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        if "entry-level" in text or "junior" in text or "fresh graduate" in text:
            return 0.0
        return None

    def _is_entry_level_jd(self, jd_profile: JobDescriptionProfile) -> bool:
        """Determine whether a JD is aimed at entry-level or fresh graduate candidates."""
        combined = f"{jd_profile.job_title or ''} {jd_profile.experience_requirement or ''}".lower()
        return any(term in combined for term in ("entry-level", "entry level", "junior", "graduate", "intern"))

    def _analyze_career_history(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
    ) -> tuple[float, list[WorkExperience], list[WorkExperience], list[WorkExperience]]:
        """
        Segment candidate career history into relevant full-time jobs, unrelated jobs,
        and internships, calculating verified technical full-time tenure.
        """
        relevant_jobs: list[WorkExperience] = []
        unrelated_jobs: list[WorkExperience] = []
        internships: list[WorkExperience] = []

        target_title = (jd_profile.job_title or "").lower()
        jd_skills_lower = {s.lower() for s in jd_profile.required_skills + jd_profile.preferred_skills}

        relevant_ft_years = 0.0
        unrelated_ft_years = 0.0

        for exp in resume_profile.experience:
            title = (exp.job_title or "").lower()
            emp_type = (exp.employment_type or "").lower()
            is_internship = ("intern" in title) or ("intern" in emp_type)

            # Check domain relevance:
            # 1. Matching title tokens or backend/software/cloud/engineer
            # 2. Overlap with JD skills in exp.technologies
            tech_overlap = any(t.lower() in jd_skills_lower for t in exp.technologies)
            title_match = bool(target_title and any(token in title for token in target_title.split() if len(token) > 3))
            is_engineering = any(kw in title for kw in ("engineer", "developer", "architect", "programmer", "specialist"))

            is_relevant = is_engineering or tech_overlap or title_match
            duration = self._calculate_job_duration_years(exp)

            if is_internship:
                internships.append(exp)
            elif is_relevant:
                relevant_jobs.append(exp)
                relevant_ft_years += duration
            else:
                unrelated_jobs.append(exp)
                unrelated_ft_years += duration

        # Precedence matrix tenure rule:
        # 1. If relevant jobs present: use relevant FT tenure (unrelated jobs ignored)
        # 2. If no relevant jobs, but candidate has relevant internships or projects:
        #    qualifies via Tier 3 or 4 -> unrelated employment tenure does not count (0.0)
        # 3. If candidate has ONLY unrelated employment (Scenario 5, Tier 5):
        #    tenure is evaluated on full-time professional employment tenure (e.g. 5 yrs as Data Analyst >= 5 yrs)
        if relevant_jobs:
            candidate_ft_years = relevant_ft_years
        elif internships or self._has_relevant_projects(resume_profile, jd_profile):
            candidate_ft_years = 0.0
        else:
            candidate_ft_years = unrelated_ft_years

        return candidate_ft_years, relevant_jobs, unrelated_jobs, internships

    def _calculate_job_duration_years(self, exp: WorkExperience) -> float:
        """Calculate duration in years from employment_start and employment_end."""
        # 1. Check if duration is explicitly stated in original text
        match_yrs = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', exp.original_text, re.IGNORECASE)
        if match_yrs:
            try:
                return float(match_yrs.group(1))
            except ValueError:
                pass

        # 2. Parse start and end years
        start_year = self._extract_year(exp.employment_start)
        end_year = self._extract_year(exp.employment_end)

        if start_year and end_year:
            diff = max(0.5, float(end_year - start_year))
            return diff
        elif start_year:
            # Current year fallback
            current_year = 2026
            return max(0.5, float(current_year - start_year))

        return 1.0  # Default conservative duration if dates unparseable

    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        s = date_str.strip().lower()
        if "present" in s or "current" in s or "now" in s:
            return 2026
        match = re.search(r'\b(20\d\d|19\d\d)\b', s)
        if match:
            return int(match.group(1))
        return None

    def _determine_precedence_tier(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
        relevant_jobs: list[WorkExperience],
        unrelated_jobs: list[WorkExperience],
        internships: list[WorkExperience],
        required_years: Optional[float],
    ) -> ExperienceTier:
        """
        Apply the approved Precedence Matrix for mixed candidate backgrounds.
        Evaluates the candidate's highest level of verified domain-relevant practical experience.
        """
        # Precedence 1: Relevant Full-Time Role Present
        if relevant_jobs:
            target_title = (jd_profile.job_title or "").lower()
            # Check for direct alignment (Tier 1) vs adjacent alignment (Tier 2)
            has_direct = False
            for rj in relevant_jobs:
                rj_title = (rj.job_title or "").lower()
                # Direct match: same title or core engineering discipline match
                if target_title and target_title in rj_title:
                    has_direct = True
                    break
                elif "backend" in target_title and "backend" in rj_title:
                    has_direct = True
                    break
                elif ("software engineer" in rj_title or "senior" in rj_title) and ("backend" in target_title or "software" in target_title):
                    has_direct = True
                    break

            if has_direct:
                return ExperienceTier.DIRECT_ALIGNMENT
            else:
                return ExperienceTier.ADJACENT_ALIGNMENT

        # Precedence 2: No Relevant FT, but Relevant Internship Present
        if internships:
            # Check if internship is technical/domain relevant
            jd_skills_lower = {s.lower() for s in jd_profile.required_skills + jd_profile.preferred_skills}
            for it in internships:
                if any(t.lower() in jd_skills_lower for t in it.technologies) or "software" in (it.job_title or "").lower() or "engineering" in (it.job_title or "").lower():
                    return ExperienceTier.RELEVANT_INTERNSHIP

        # Precedence 3: No Relevant Jobs/Internships, but Verified Relevant Projects Present
        has_relevant_projects = self._has_relevant_projects(resume_profile, jd_profile)
        if has_relevant_projects:
            return ExperienceTier.APPLIED_PROJECTS

        # Precedence 4: Only Non-Technical Employment (Unrelated Background)
        if unrelated_jobs:
            return ExperienceTier.UNRELATED_BACKGROUND

        # Precedence 5: No Employment and No Relevant Projects
        if not resume_profile.experience and not resume_profile.projects:
            return ExperienceTier.UNKNOWN

        return ExperienceTier.UNRELATED_BACKGROUND

    def _has_relevant_projects(self, resume_profile: ResumeProfile, jd_profile: JobDescriptionProfile) -> bool:
        """Check whether candidate has verified relevant engineering projects matching JD criteria."""
        jd_skills_lower = {s.lower() for s in jd_profile.required_skills + jd_profile.preferred_skills}
        for proj in resume_profile.projects:
            if any(t.lower() in jd_skills_lower for t in proj.technologies):
                return True
            for sk in jd_skills_lower:
                if sk in proj.description.lower():
                    return True
        return False
