"""
Unit tests for core/matching/education_evaluator.py.
Verifies generic vs. discipline-specific evaluation, major relevance scoring,
degree rank parsing, and dimension neutralization when education is unstated in JD.
"""
import pytest

from core.matching.education_evaluator import EducationEvaluator
from models.matching import EvidenceStatus, JobDescriptionProfile
from models.resume_entities import EducationRecord
from models.resume_profile import ResumeProfile


def make_edu_resume(records: list[EducationRecord]) -> ResumeProfile:
    return ResumeProfile(
        filename="edu_resume.pdf",
        education=records,
    )


class TestEducationEvaluator:
    """Tests deterministic education scoring rules and discipline relevance."""

    @pytest.fixture
    def evaluator(self) -> EducationEvaluator:
        return EducationEvaluator()

    def test_generic_bachelor_requirement_accepts_any_degree_field(self, evaluator: EducationEvaluator):
        # JD requires generic Bachelor's (no discipline specified)
        # Candidate holds a Bachelor's in Economics (non-STEM)
        # Rule: Major relevance is NOT evaluated; awarded 100.0%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Arts",
                    field_of_study="Economics",
                    institution="State University",
                    original_text="Bachelor of Arts in Economics",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's degree required",
            original_description="Must hold a Bachelor's degree",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["dimension_score"] == 100.0
        assert result["level_score"] == 100.0
        assert result["major_score"] is None
        assert result["is_active"] is True

    def test_discipline_specific_exact_core_stem(self, evaluator: EducationEvaluator):
        # JD requires Bachelor's in CS or related STEM
        # Candidate holds B.S. in Computer Science -> 100% level, 100% major -> 100%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    institution="Tech University",
                    original_text="B.S. in Computer Science",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's degree in Computer Science or related STEM field",
            original_description="Degree in Computer Science or STEM",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["dimension_score"] == 100.0
        assert result["level_score"] == 100.0
        assert result["major_score"] == 100.0

    def test_discipline_specific_adjacent_stem_scoring(self, evaluator: EducationEvaluator):
        # Scenario 2: Candidate holds degree in Information Systems (Adjacent STEM = 75%)
        # Score = 0.60(100.0) + 0.40(75.0) = 90.0%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Information Systems",
                    institution="Tech University",
                    original_text="B.S. in Information Systems",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's in Computer Science or related STEM field",
            original_description="Degree in CS or related STEM field",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["level_score"] == 100.0
        assert result["major_score"] == 75.0
        assert pytest.approx(result["dimension_score"], rel=1e-3) == 90.0

    def test_discipline_specific_non_stem_major_scoring(self, evaluator: EducationEvaluator):
        # Candidate holds Bachelor's in English Literature when CS/STEM is explicitly required
        # Score = 0.60(100.0) + 0.40(25.0) = 70.0%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Arts",
                    field_of_study="English Literature",
                    institution="City College",
                    original_text="B.A. in English Literature",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's in Computer Science or related STEM discipline",
            original_description="CS or STEM degree required",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["level_score"] == 100.0
        assert result["major_score"] == 25.0
        assert pytest.approx(result["dimension_score"], rel=1e-3) == 70.0

    def test_major_unspecified_in_resume_neutralizes_major_subcomponent(self, evaluator: EducationEvaluator):
        # Candidate degree is simply "Bachelor of Science" without major
        # Major is UNKNOWN_EVIDENCE -> neutralized, evaluated on Level = 100%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study=None,
                    institution="University",
                    original_text="Bachelor of Science",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's degree in Computer Science required",
            original_description="CS degree required",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["level_score"] == 100.0
        assert result["major_score"] is None
        assert result["dimension_score"] == 100.0

    def test_lower_degree_level_receives_partial_credit(self, evaluator: EducationEvaluator):
        # JD requires Master's, candidate holds Bachelor's (one rank below)
        # Level = 50.0%
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    institution="University",
                    original_text="B.S. in Computer Science",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement="Master's degree required",
            original_description="Master's required",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["level_score"] == 50.0
        assert result["dimension_score"] == 50.0

    def test_no_education_in_jd_neutralizes_dimension(self, evaluator: EducationEvaluator):
        # Scenario 3: JD states no education requirement
        resume = make_edu_resume(
            [
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    institution="University",
                    original_text="B.S. in Computer Science",
                )
            ]
        )
        jd = JobDescriptionProfile(
            education_requirement=None,
            original_description="No degree required",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["is_active"] is False
        assert result["dimension_score"] == 0.0
        assert "no academic education criteria" in (result["neutralization_explanation"] or "").lower()

    def test_missing_education_in_resume(self, evaluator: EducationEvaluator):
        # JD requires degree, resume has zero education records
        resume = make_edu_resume([])
        jd = JobDescriptionProfile(
            education_requirement="Bachelor's degree required",
            original_description="Must have degree",
        )

        result = evaluator.evaluate(resume, jd)
        assert result["is_active"] is True
        assert result["dimension_score"] == 0.0
        assert result["evidence_status"] == EvidenceStatus.MISSING_EVIDENCE

