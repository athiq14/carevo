"""
Unit tests for core/matching/skill_matcher.py.
Verifies exact match, synonym resolution, related technology credit,
URL token exclusion, priority weighting, and duplicate non-inflation.
"""
import pytest

from core.matching.skill_matcher import SkillMatcher
from models.matching import (
    EvidenceStatus,
    JobDescriptionProfile,
    JobRequirement,
    RequirementCategory,
    RequirementPriority,
    SkillMatchType,
)
from models.resume_entities import ExtractedSkill, ProjectRecord, SkillCategory, WorkExperience
from models.resume_profile import ResumeProfile


def make_test_resume(
    skills: list[str],
    exp_techs: list[str] | None = None,
    proj_techs: list[str] | None = None,
    raw_evidences: dict[str, str] | None = None,
) -> ResumeProfile:
    """Helper to assemble a synthetic ResumeProfile for testing."""
    raw_ev = raw_evidences or {}
    skill_objs = [
        ExtractedSkill(
            name=s,
            original_mention=s,
            category=SkillCategory.PROGRAMMING_LANGUAGES,
            raw_evidence=raw_ev.get(s, f"Proficient in {s}"),
        )
        for s in skills
    ]
    exp_objs = []
    if exp_techs:
        exp_objs.append(
            WorkExperience(
                job_title="Software Engineer",
                company="TechCorp",
                technologies=exp_techs,
                original_text="Engineered services using " + ", ".join(exp_techs),
            )
        )
    proj_objs = []
    if proj_techs:
        proj_objs.append(
            ProjectRecord(
                project_name="Web App",
                technologies=proj_techs,
                original_text="Built project using " + ", ".join(proj_techs),
            )
        )
    return ResumeProfile(
        filename="test_resume.pdf",
        skills=skill_objs,
        experience=exp_objs,
        projects=proj_objs,
    )


def make_test_jd(
    required_skills: list[str],
    preferred_skills: list[str] | None = None,
) -> JobDescriptionProfile:
    """Helper to assemble a synthetic JobDescriptionProfile for testing."""
    reqs = []
    pref = preferred_skills or []
    for s in required_skills:
        reqs.append(
            JobRequirement(
                requirement_id=f"req-{s.lower()}",
                requirement_text=f"Must have {s}",
                canonical_name=s,
                category=RequirementCategory.TECHNICAL_SKILL,
                priority=RequirementPriority.REQUIRED,
            )
        )
    for s in pref:
        reqs.append(
            JobRequirement(
                requirement_id=f"pref-{s.lower()}",
                requirement_text=f"Preferred {s}",
                canonical_name=s,
                category=RequirementCategory.TECHNICAL_SKILL,
                priority=RequirementPriority.PREFERRED,
            )
        )
    return JobDescriptionProfile(
        original_description="Job Description Text",
        required_skills=required_skills,
        preferred_skills=pref,
        normalized_requirements=reqs,
    )


class TestSkillMatcher:
    """Tests deterministic skill matching precision and scoring formulas."""

    @pytest.fixture
    def matcher(self) -> SkillMatcher:
        return SkillMatcher()

    def test_exact_match_awarded_full_credit(self, matcher: SkillMatcher):
        resume = make_test_resume(skills=["Python", "FastAPI"])
        jd = make_test_jd(required_skills=["Python", "FastAPI"])

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 100.0
        assert len(result["matched_skills"]) == 2
        assert all(m.match_type == SkillMatchType.EXACT_MATCH for m in result["matched_skills"])

    def test_recognized_synonym_awarded_full_credit(self, matcher: SkillMatcher):
        # Candidate lists "Postgres" and "K8s"; JD asks for "PostgreSQL" and "Kubernetes"
        resume = make_test_resume(skills=["Postgres", "K8s"])
        jd = make_test_jd(required_skills=["PostgreSQL", "Kubernetes"])

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 100.0
        assert len(result["matched_skills"]) == 2
        assert all(m.match_type == SkillMatchType.SYNONYM_MATCH for m in result["matched_skills"])

    def test_related_technology_awarded_approved_partial_credit(self, matcher: SkillMatcher):
        # Candidate has Flask (mapped neighbor to FastAPI with factor 0.50)
        resume = make_test_resume(skills=["Flask"])
        jd = make_test_jd(required_skills=["FastAPI"])

        result = matcher.evaluate(resume, jd)
        # Denominator = 1.0 (1 req skill). Numerator = 0.50. Score = 50.0%.
        assert result["dimension_score"] == 50.0
        assert len(result["partial_matches"]) == 1
        pm = result["partial_matches"][0]
        assert pm.match_type == SkillMatchType.RELATED_TECHNOLOGY
        assert pm.evidence_status == EvidenceStatus.PARTIAL_EVIDENCE
        assert pm.match_confidence == 0.50

    def test_unmapped_technology_in_same_domain_receives_zero_credit(self, matcher: SkillMatcher):
        # Candidate has Ruby on Rails (not a mapped neighbor to FastAPI in RELATED_TECH_MAP)
        resume = make_test_resume(skills=["Ruby"])
        jd = make_test_jd(required_skills=["FastAPI"])

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 0.0
        assert len(result["missing_evidence"]) == 1
        assert result["missing_evidence"][0].skill_name == "FastAPI"

    def test_required_and_preferred_weighting(self, matcher: SkillMatcher):
        # Scenario 1 skill calculation:
        # Req: Python (1.0), Postgres (1.0), FastAPI (1.0) -> 3.0 / 3.0
        # Pref: Docker (0.5), AWS (0.0) -> 0.5 / 1.0
        # Score = (3.0 + 0.5) / (3.0 + 1.0) = 3.5 / 4.0 = 87.5%
        resume = make_test_resume(skills=["Python", "PostgreSQL", "FastAPI", "Docker"])
        jd = make_test_jd(
            required_skills=["Python", "PostgreSQL", "FastAPI"],
            preferred_skills=["Docker", "AWS"],
        )

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 87.5
        assert len(result["matched_skills"]) == 4  # Python, Postgres, FastAPI, Docker
        assert len(result["missing_evidence"]) == 1  # AWS

    def test_duplicate_skills_across_sections_do_not_inflate_score(self, matcher: SkillMatcher):
        # Candidate lists Python in Skills, Experience, and Projects
        resume = make_test_resume(
            skills=["Python"],
            exp_techs=["Python"],
            proj_techs=["Python"],
        )
        jd = make_test_jd(required_skills=["Python", "Go"])

        result = matcher.evaluate(resume, jd)
        # Denominator = 2.0. Python credit is strictly 1.0 (not 3.0). Score = 50.0%.
        assert result["dimension_score"] == 50.0

    def test_technologies_in_urls_do_not_earn_skill_credit(self, matcher: SkillMatcher):
        # URL evidence should not create false skill token credit
        resume = make_test_resume(
            skills=["Python"],
            raw_evidences={"Python": "https://github.com/user/docker-kubernetes-project"},
        )
        jd = make_test_jd(required_skills=["Docker", "Kubernetes"])

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 0.0
        assert len(result["missing_evidence"]) == 2

    def test_empty_jd_skills_neutralizes_dimension(self, matcher: SkillMatcher):
        resume = make_test_resume(skills=["Python"])
        jd = make_test_jd(required_skills=[], preferred_skills=[])

        result = matcher.evaluate(resume, jd)
        assert result["is_active"] is False
        assert result["dimension_score"] == 0.0
        assert "no technical skill requirements" in (result["neutralization_explanation"] or "").lower()

    def test_unknown_required_skill_denominator_neutralization(self, matcher: SkillMatcher):
        # Req: Python (matched), Leadership (qualitative -> UNKNOWN)
        # Pref: Docker (missing technical skill -> penalized)
        # Denominator: Python (1.0) + Docker (0.5) = 1.5 (Leadership's 1.0 is neutralized)
        # Numerator: Python (1.0) + Docker (0.0) = 1.0
        # Score = 1.0 / 1.5 = 66.67%
        resume = make_test_resume(skills=["Python"])
        jd = make_test_jd(
            required_skills=["Python", "Leadership"],
            preferred_skills=["Docker"],
        )

        result = matcher.evaluate(resume, jd)
        assert round(result["dimension_score"], 2) == 66.67
        assert len(result["matched_skills"]) == 1
        assert len(result["unknown_evidence"]) == 1
        assert result["unknown_evidence"][0].skill_name == "Leadership"
        assert result["unknown_evidence"][0].match_type == SkillMatchType.UNKNOWN
        assert len(result["missing_evidence"]) == 1
        assert result["missing_evidence"][0].skill_name == "Docker"
        assert result["missing_evidence"][0].evidence_status == EvidenceStatus.MISSING_EVIDENCE

    def test_unknown_preferred_skill_denominator_neutralization(self, matcher: SkillMatcher):
        # Req: Python (matched) -> 1.0 / 1.0
        # Pref: Docker (matched) -> 0.5 / 0.5, Team Player (qualitative -> UNKNOWN)
        # Denominator: Python (1.0) + Docker (0.5) = 1.5 (Team Player's 0.5 is neutralized)
        # Numerator: Python (1.0) + Docker (0.5) = 1.5
        # Score = 1.5 / 1.5 = 100.0%
        resume = make_test_resume(skills=["Python", "Docker"])
        jd = make_test_jd(
            required_skills=["Python"],
            preferred_skills=["Docker", "Team Player"],
        )

        result = matcher.evaluate(resume, jd)
        assert result["dimension_score"] == 100.0
        assert len(result["matched_skills"]) == 2
        assert len(result["unknown_evidence"]) == 1
        assert result["unknown_evidence"][0].skill_name == "Team Player"

    def test_missing_technical_skills_remain_penalized(self, matcher: SkillMatcher):
        # Normal technical skills absent from resume MUST remain MISSING_EVIDENCE and penalize score
        resume = make_test_resume(skills=["Python"])
        jd = make_test_jd(
            required_skills=["Python", "Kubernetes"],
            preferred_skills=["Go"],
        )

        result = matcher.evaluate(resume, jd)
        # Denominator: 1.0 (Python) + 1.0 (Kubernetes) + 0.5 (Go) = 2.5
        # Numerator: 1.0 (Python)
        # Score = 1.0 / 2.5 = 40.0%
        assert result["dimension_score"] == 40.0
        assert len(result["matched_skills"]) == 1
        assert len(result["missing_evidence"]) == 2
        assert {g.skill_name for g in result["missing_evidence"]} == {"Kubernetes", "Go"}
        assert len(result["unknown_evidence"]) == 0

    def test_all_skills_unknown_neutralizes_dimension(self, matcher: SkillMatcher):
        # When all skills are qualitative/unassessable, remaining denominator is 0.0:
        # Dimension is neutralized, score is 0.0 (NOT artificial 100.0%), status is UNKNOWN_EVIDENCE
        resume = make_test_resume(skills=["Python", "PostgreSQL"])
        jd = make_test_jd(
            required_skills=["Leadership"],
            preferred_skills=["Communication Skills"],
        )

        result = matcher.evaluate(resume, jd)
        assert result["is_active"] is False
        assert result["dimension_score"] == 0.0
        assert result["evidence_status"] == EvidenceStatus.UNKNOWN_EVIDENCE
        assert len(result["unknown_evidence"]) == 2
        assert "neutralized" in (result["neutralization_explanation"] or "").lower()

