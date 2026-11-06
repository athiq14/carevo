"""
Deterministic skill matching engine.
Compares candidate resume evidence against job description requirements using a strict
5-tier precision taxonomy: EXACT_MATCH, SYNONYM_MATCH, RELATED_TECHNOLOGY, NO_MATCH, UNKNOWN.
Enforces required vs. preferred priority weighting and denies artificial inflation from duplicates or URLs.
"""
import re
from typing import Optional

from models.matching import (
    EvidenceStatus,
    JobDescriptionProfile,
    JobRequirement,
    RequirementCategory,
    RequirementPriority,
    SkillGap,
    SkillMatch,
    SkillMatchType,
)
from models.resume_profile import ResumeProfile
from processors.entity_extractor import SKILL_TAXONOMY, URL_EXCLUSION_PATTERN
from processors.jd_processor import SYNONYM_LOOKUP


# ---------------------------------------------------------------------------
# Explicit Curated Architectural Adjacency Map
# ---------------------------------------------------------------------------
# Strict mappings between related technologies. Unmapped tools in the same
# domain receive 0.0 credit to prevent false equivalence.
RELATED_TECH_MAP: dict[str, dict[str, float]] = {
    "FastAPI": {"Flask": 0.50, "Django": 0.40, "Starlette": 0.60},
    "Flask": {"FastAPI": 0.50, "Django": 0.40},
    "Django": {"Flask": 0.40, "FastAPI": 0.40, "Ruby on Rails": 0.30},
    "React": {"Next.js": 0.60, "Vue.js": 0.40, "Angular": 0.40},
    "Next.js": {"React": 0.60, "Vue.js": 0.40},
    "Angular": {"React": 0.40, "Vue.js": 0.40},
    "Vue.js": {"React": 0.40, "Angular": 0.40},
    "PostgreSQL": {"MySQL": 0.50, "Oracle Database": 0.40, "Microsoft SQL Server": 0.40, "SQLite": 0.40},
    "MySQL": {"PostgreSQL": 0.50, "Microsoft SQL Server": 0.40, "Oracle Database": 0.40, "SQLite": 0.40},
    "Kafka": {"RabbitMQ": 0.40, "AWS SQS": 0.35, "Apache Pulsar": 0.50},
    "RabbitMQ": {"Kafka": 0.40, "AWS SQS": 0.35},
    "AWS": {"Google Cloud Platform": 0.50, "Microsoft Azure": 0.50},
    "Google Cloud Platform": {"AWS": 0.50, "Microsoft Azure": 0.50},
    "Microsoft Azure": {"AWS": 0.50, "Google Cloud Platform": 0.50},
    "PyTorch": {"TensorFlow": 0.50, "Keras": 0.45},
    "TensorFlow": {"PyTorch": 0.50, "Keras": 0.45},
}


# Set of recognized qualitative, subjective, or non-technical traits
QUALITATIVE_OR_UNASSESSABLE_SKILLS = {
    "team player",
    "teamwork",
    "communication",
    "communication skills",
    "verbal communication",
    "written communication",
    "leadership",
    "leadership potential",
    "mentorship",
    "problem solving",
    "critical thinking",
    "analytical thinking",
    "analytical mindset",
    "work ethic",
    "strong work ethic",
    "fast learner",
    "quick learner",
    "adaptability",
    "interpersonal skills",
    "time management",
    "collaboration",
    "collaborative",
    "attention to detail",
    "creativity",
    "self-motivated",
    "passion",
    "culture fit",
    "initiative",
    "conflict resolution",
    "emotional intelligence",
}


class SkillMatcher:
    """
    Deterministic evaluator for technical skill compatibility.
    Calculates exact, synonym, and pre-mapped related technology alignment.
    """

    def __init__(
        self,
        synonym_lookup: dict[str, str] = SYNONYM_LOOKUP,
        related_tech_map: dict[str, dict[str, float]] = RELATED_TECH_MAP,
    ) -> None:
        self.synonym_lookup = synonym_lookup
        self.related_tech_map = related_tech_map

    def evaluate(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
    ) -> dict[str, object]:
        """
        Evaluate candidate technical skills against the job description.

        Returns:
            dict containing:
                - dimension_score: float bounded [0.0, 100.0]
                - matched_skills: list[SkillMatch] (EXACT_MATCH, SYNONYM_MATCH)
                - partial_matches: list[SkillMatch] (RELATED_TECHNOLOGY)
                - missing_evidence: list[SkillGap] (NO_MATCH)
                - unknown_evidence: list[SkillMatch]
                - is_active: bool
                - neutralization_explanation: Optional[str]
        """
        # Step 1: Collect and canonicalize all candidate skill evidence from resume
        candidate_skills = self._extract_candidate_skills(resume_profile)

        # Step 2: Determine unique JD skills (Required vs Preferred)
        # Required skills take precedence over preferred if mentioned in both
        req_skills: list[str] = []
        seen_req: set[str] = set()
        for s in jd_profile.required_skills:
            canon = self.synonym_lookup.get(s.strip().lower(), s.strip())
            if canon.lower() not in seen_req:
                seen_req.add(canon.lower())
                req_skills.append(canon)

        pref_skills: list[str] = []
        for s in jd_profile.preferred_skills:
            canon = self.synonym_lookup.get(s.strip().lower(), s.strip())
            if canon.lower() not in seen_req and canon.lower() not in [p.lower() for p in pref_skills]:
                pref_skills.append(canon)

        total_jd_skills = len(req_skills) + len(pref_skills)
        if total_jd_skills == 0:
            return {
                "dimension_score": 0.0,
                "matched_skills": [],
                "partial_matches": [],
                "missing_evidence": [],
                "unknown_evidence": [],
                "is_active": False,
                "evidence_status": EvidenceStatus.UNKNOWN_EVIDENCE,
                "neutralization_explanation": "Job description specifies no technical skill requirements.",
            }

        matched_skills: list[SkillMatch] = []
        partial_matches: list[SkillMatch] = []
        missing_evidence: list[SkillGap] = []
        unknown_evidence: list[SkillMatch] = []

        total_numerator = 0.0
        total_denominator = 0.0

        # Step 3: Evaluate Required Skills (Weight = 1.0)
        for skill_name in req_skills:
            match_res, credit = self._match_individual_skill(
                skill_name=skill_name,
                priority=RequirementPriority.REQUIRED,
                candidate_skills=candidate_skills,
                jd_profile=jd_profile,
            )

            if match_res.match_type in (SkillMatchType.EXACT_MATCH, SkillMatchType.SYNONYM_MATCH):
                total_numerator += credit * 1.0
                total_denominator += 1.0
                matched_skills.append(match_res)
            elif match_res.match_type == SkillMatchType.RELATED_TECHNOLOGY:
                total_numerator += credit * 1.0
                total_denominator += 1.0
                partial_matches.append(match_res)
            elif match_res.match_type == SkillMatchType.UNKNOWN:
                # UNKNOWN: requirement weight is removed from denominator (neutralized)
                unknown_evidence.append(match_res)
            else:
                # NO_MATCH / MISSING_EVIDENCE: zero numerator, retain weight in denominator
                total_denominator += 1.0
                gap = SkillGap(
                    skill_name=skill_name,
                    priority=RequirementPriority.REQUIRED,
                    evidence_status=EvidenceStatus.MISSING_EVIDENCE,
                    gap_explanation=f"Required skill '{skill_name}' was not evidenced in candidate resume skills, experience, or projects.",
                    suggested_learning_direction=f"Gain hands-on practical proficiency with {skill_name}.",
                )
                missing_evidence.append(gap)

        # Step 4: Evaluate Preferred Skills (Weight = 0.5)
        for skill_name in pref_skills:
            match_res, credit = self._match_individual_skill(
                skill_name=skill_name,
                priority=RequirementPriority.PREFERRED,
                candidate_skills=candidate_skills,
                jd_profile=jd_profile,
            )

            if match_res.match_type in (SkillMatchType.EXACT_MATCH, SkillMatchType.SYNONYM_MATCH):
                total_numerator += credit * 0.5
                total_denominator += 0.5
                matched_skills.append(match_res)
            elif match_res.match_type == SkillMatchType.RELATED_TECHNOLOGY:
                total_numerator += credit * 0.5
                total_denominator += 0.5
                partial_matches.append(match_res)
            elif match_res.match_type == SkillMatchType.UNKNOWN:
                # UNKNOWN: requirement weight is removed from denominator (neutralized)
                unknown_evidence.append(match_res)
            else:
                # NO_MATCH / MISSING_EVIDENCE: zero numerator, retain weight in denominator
                total_denominator += 0.5
                gap = SkillGap(
                    skill_name=skill_name,
                    priority=RequirementPriority.PREFERRED,
                    evidence_status=EvidenceStatus.MISSING_EVIDENCE,
                    gap_explanation=f"Preferred skill '{skill_name}' was not evidenced in candidate profile.",
                    suggested_learning_direction=f"Explore foundational concepts and tools in {skill_name}.",
                )
                missing_evidence.append(gap)

        # Step 5: Check if remaining denominator is zero
        if total_denominator <= 0.0:
            return {
                "dimension_score": 0.0,
                "matched_skills": matched_skills,
                "partial_matches": partial_matches,
                "missing_evidence": missing_evidence,
                "unknown_evidence": unknown_evidence,
                "is_active": False,
                "evidence_status": EvidenceStatus.UNKNOWN_EVIDENCE,
                "neutralization_explanation": (
                    "All specified technical skill requirements are unassessed or qualitative (remaining denominator is zero); Skills dimension neutralized."
                ),
            }

        dimension_score = (total_numerator / total_denominator) * 100.0
        dimension_score = max(0.0, min(100.0, dimension_score))
        ev_status = (
            EvidenceStatus.VERIFIED_EVIDENCE
            if dimension_score >= 80.0
            else (EvidenceStatus.PARTIAL_EVIDENCE if dimension_score > 0.0 else EvidenceStatus.MISSING_EVIDENCE)
        )

        return {
            "dimension_score": dimension_score,
            "matched_skills": matched_skills,
            "partial_matches": partial_matches,
            "missing_evidence": missing_evidence,
            "unknown_evidence": unknown_evidence,
            "is_active": True,
            "evidence_status": ev_status,
            "neutralization_explanation": None,
        }

    def _extract_candidate_skills(self, resume_profile: ResumeProfile) -> dict[str, tuple[str, list[str]]]:
        """
        Extract and canonicalize all candidate skills with their textual evidence.
        Returns mapping from canonical skill name to (original_mention, list of evidence snippets).
        URL strings are strictly excluded to prevent false token credit.
        """
        candidate_skills: dict[str, tuple[str, list[str]]] = {}

        # 1. From structured extracted skills
        for sk in resume_profile.skills:
            clean_mention = sk.name.strip()
            canon = self.synonym_lookup.get(clean_mention.lower(), clean_mention)
            evidence = sk.raw_evidence or sk.original_mention
            # Exclude evidence that consists exclusively of URLs
            if URL_EXCLUSION_PATTERN.fullmatch(evidence.strip()):
                continue
            if canon not in candidate_skills:
                candidate_skills[canon] = (clean_mention, [])
            candidate_skills[canon][1].append(f"Skills section: {evidence.strip()}")

        # 2. From work experience technologies and responsibilities
        for exp in resume_profile.experience:
            for tech in exp.technologies:
                clean_tech = tech.strip()
                canon = self.synonym_lookup.get(clean_tech.lower(), clean_tech)
                role_label = f"{exp.job_title or 'Role'} at {exp.company or 'Company'}"
                if canon not in candidate_skills:
                    candidate_skills[canon] = (clean_tech, [])
                candidate_skills[canon][1].append(f"Work experience ({role_label})")

        # 3. From projects technologies
        for proj in resume_profile.projects:
            for tech in proj.technologies:
                clean_tech = tech.strip()
                canon = self.synonym_lookup.get(clean_tech.lower(), clean_tech)
                if canon not in candidate_skills:
                    candidate_skills[canon] = (clean_tech, [])
                candidate_skills[canon][1].append(f"Project '{proj.project_name}'")

        return candidate_skills

    def _is_unassessable_skill(
        self,
        skill_name: str,
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> bool:
        """
        Determine if a skill requirement is qualitative/unassessable from resume evidence.
        Technical skills absent from resume are MISSING_EVIDENCE, not UNKNOWN.
        """
        clean = skill_name.strip().lower()
        if clean in QUALITATIVE_OR_UNASSESSABLE_SKILLS:
            return True
        if any(clean.startswith(prefix) for prefix in ("qualitative:", "unassessed:", "soft skill:", "unknown:")):
            return True
        if jd_profile and jd_profile.normalized_requirements:
            for req in jd_profile.normalized_requirements:
                if req.canonical_name.lower() == clean or req.requirement_text.lower() == clean:
                    if req.category in (RequirementCategory.RESPONSIBILITY, RequirementCategory.ROLE_RELEVANCE):
                        return True
        return False

    def _match_individual_skill(
        self,
        skill_name: str,
        priority: RequirementPriority,
        candidate_skills: dict[str, tuple[str, list[str]]],
        jd_profile: Optional[JobDescriptionProfile] = None,
    ) -> tuple[SkillMatch, float]:
        """
        Match a single skill requirement against candidate evidence.
        Returns (SkillMatch, credit) where credit is in [0.0, 1.0].
        """
        canonical_target = self.synonym_lookup.get(skill_name.lower(), skill_name)

        # 0. Check for unassessable / qualitative requirement (UNKNOWN)
        if self._is_unassessable_skill(skill_name, jd_profile):
            return (
                SkillMatch(
                    skill_name=skill_name,
                    canonical_name=canonical_target,
                    requirement_priority=priority,
                    match_type=SkillMatchType.UNKNOWN,
                    evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                    resume_evidence_references=[],
                    jd_evidence_references=[f"{priority.value}: {skill_name}"],
                    match_confidence=0.0,
                    explanation=(
                        f"Requirement '{skill_name}' is qualitative or cannot be evaluated from resume evidence; "
                        "neutralized from technical skill denominator."
                    ),
                ),
                0.0,
            )

        # 1. Check for Exact Match or Direct Canonical Match
        for cand_canon, (orig_mention, ev_list) in candidate_skills.items():
            if cand_canon.lower() == canonical_target.lower():
                is_exact = (orig_mention.strip().lower() == skill_name.strip().lower())
                match_type = SkillMatchType.EXACT_MATCH if is_exact else SkillMatchType.SYNONYM_MATCH
                return (
                    SkillMatch(
                        skill_name=skill_name,
                        canonical_name=canonical_target,
                        requirement_priority=priority,
                        match_type=match_type,
                        evidence_status=EvidenceStatus.VERIFIED_EVIDENCE,
                        resume_evidence_references=ev_list[:3],
                        jd_evidence_references=[f"{priority.value}: {skill_name}"],
                        match_confidence=1.0,
                        explanation=f"Candidate demonstrates verified proficiency in {canonical_target}.",
                    ),
                    1.0,
                )

        # 2. Check for Pre-Mapped Related Technology
        neighbors = self.related_tech_map.get(canonical_target, {})
        for cand_canon, (orig_mention, ev_list) in candidate_skills.items():
            for neighbor_name, factor in neighbors.items():
                if cand_canon.lower() == neighbor_name.lower():
                    return (
                        SkillMatch(
                            skill_name=skill_name,
                            canonical_name=canonical_target,
                            requirement_priority=priority,
                            match_type=SkillMatchType.RELATED_TECHNOLOGY,
                            evidence_status=EvidenceStatus.PARTIAL_EVIDENCE,
                            resume_evidence_references=ev_list[:3],
                            jd_evidence_references=[f"{priority.value}: {skill_name}"],
                            match_confidence=factor,
                            explanation=(
                                f"Candidate demonstrates exposure to {neighbor_name}, a recognized related "
                                f"technology to requested {canonical_target} (credit factor: {factor:.2f})."
                            ),
                        ),
                        factor,
                    )

        # 3. No match found
        return (
            SkillMatch(
                skill_name=skill_name,
                canonical_name=canonical_target,
                requirement_priority=priority,
                match_type=SkillMatchType.NO_MATCH,
                evidence_status=EvidenceStatus.MISSING_EVIDENCE,
                resume_evidence_references=[],
                jd_evidence_references=[f"{priority.value}: {skill_name}"],
                match_confidence=0.0,
                explanation=f"Requirement '{skill_name}' was not evidenced in candidate resume.",
            ),
            0.0,
        )
