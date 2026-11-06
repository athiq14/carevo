"""
Unit tests for core/matching/experience_evaluator.py.
Verifies tenure calculation, discrete experience tiers, entry-level special rubrics,
precedence matrix resolution for mixed backgrounds, and sub-component neutralization.
"""
import pytest

from core.matching.experience_evaluator import ExperienceEvaluator
from models.matching import (
    ExperienceTier,
    JobDescriptionProfile,
)
from models.resume_entities import ProjectRecord, WorkExperience
from models.resume_profile import ResumeProfile


def make_exp_resume(
    jobs: list[WorkExperience],
    projects: list[ProjectRecord] | None = None,
) -> ResumeProfile:
    return ResumeProfile(
        filename="exp_resume.pdf",
        experience=jobs,
        projects=projects or [],
    )


class TestExperienceEvaluator:
    """Tests deterministic experience tenure and role relevance evaluation."""

    @pytest.fixture
    def evaluator(self) -> ExperienceEvaluator:
        return ExperienceEvaluator()

    def test_direct_alignment_with_sufficient_tenure_awarded_full_score(self, evaluator: ExperienceEvaluator):
        # Scenario 1: Backend Developer with 4 years FT experience applying for 3-year Backend role
        jobs = [
            WorkExperience(
                job_title="Senior Backend Developer",
                company="TechCorp",
                employment_start="2020",
                employment_end="2024",
                employment_type="Full-time",
                technologies=["Python", "FastAPI", "PostgreSQL"],
                original_text="4 years experience building asynchronous backend services.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Backend Developer",
            experience_requirement="3+ years of experience in backend development",
            original_description="Job Description",
            required_skills=["Python", "FastAPI"],
        )

        result = evaluator.evaluate(resume, jd)
        assert result["tenure_score"] == 100.0
        assert result["relevance_score"] == 100.0
        assert result["assigned_tier"] == ExperienceTier.DIRECT_ALIGNMENT
        assert result["dimension_score"] == 100.0

    def test_partial_tenure_calculation(self, evaluator: ExperienceEvaluator):
        # Scenario 2: 2.5 years experience applying for 4-year Senior role
        # Tenure = 2.5 / 4.0 = 62.5%
        # Relevance: Junior Cloud role for Senior Cloud role -> Tier 2 (75.0%)
        # Score = 0.50 * 62.5 + 0.50 * 75.0 = 68.75%
        jobs = [
            WorkExperience(
                job_title="Junior Cloud Engineer",
                company="CloudStart",
                employment_start="2021",
                employment_end="2023",
                original_text="2.5 years of experience deploying cloud resources.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Senior Cloud Engineer",
            experience_requirement="4+ years of cloud infrastructure experience",
            original_description="Senior Cloud role",
            required_skills=["Python", "Kubernetes"],
        )

        result = evaluator.evaluate(resume, jd)
        assert pytest.approx(result["tenure_score"], rel=1e-3) == 62.5
        assert result["relevance_score"] == 75.0
        assert result["assigned_tier"] == ExperienceTier.ADJACENT_ALIGNMENT
        assert pytest.approx(result["dimension_score"], rel=1e-3) == 68.75

    def test_internships_do_not_count_toward_full_time_tenure(self, evaluator: ExperienceEvaluator):
        # Candidate has only a 6-month internship, applying for a 2-year role
        jobs = [
            WorkExperience(
                job_title="Software Engineering Intern",
                company="Acme",
                employment_start="2023",
                employment_end="2023",
                employment_type="Internship",
                original_text="6 months internship building APIs.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Software Developer",
            experience_requirement="2 years experience required",
            original_description="Role requires 2 years",
            required_skills=["Python"],
        )

        result = evaluator.evaluate(resume, jd)
        # Full-time tenure is 0.0 because internships do not count as FT tenure
        assert result["candidate_tenure_years"] == 0.0
        assert result["tenure_score"] == 0.0
        assert result["assigned_tier"] == ExperienceTier.RELEVANT_INTERNSHIP
        # Relevance for standard role is 60.0%
        assert result["relevance_score"] == 60.0
        assert result["dimension_score"] == 0.50 * 0.0 + 0.50 * 60.0  # 30.0%

    def test_entry_level_special_rubric_awards_eighty_percent_for_internship(self, evaluator: ExperienceEvaluator):
        # Scenario 4: Fresh graduate with internship applying for Entry-Level role with NO tenure required
        jobs = [
            WorkExperience(
                job_title="Software Development Intern",
                company="StartupX",
                employment_type="Internship",
                technologies=["Java", "Spring Boot"],
                original_text="Internship working on Java Spring Boot services.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Junior Software Engineer",
            experience_requirement="Entry-Level (0 years required)",
            original_description="Entry-Level software engineer role",
            required_skills=["Java", "Spring Boot"],
        )

        result = evaluator.evaluate(resume, jd)
        # Tenure is neutralized because req is 0 years
        assert result["tenure_score"] is None
        # Entry-level special rubric awards 80.0% relevance for relevant internship
        assert result["relevance_score"] == 80.0
        assert result["assigned_tier"] == ExperienceTier.RELEVANT_INTERNSHIP
        assert result["dimension_score"] == 80.0

    def test_unrelated_background_and_precedence(self, evaluator: ExperienceEvaluator):
        # Scenario 5: Data Analyst applying for Senior Infrastructure Engineer with 5 years experience
        # Tenure = 5 >= 5 = 100%
        # Relevance: Data Analyst is Tier 5 (Unrelated) = 0.0%
        # Score = 0.50 * 100 + 0.50 * 0 = 50.0%
        jobs = [
            WorkExperience(
                job_title="Data Analyst",
                company="RetailCorp",
                employment_start="2019",
                employment_end="2024",
                original_text="5 years of experience building Tableau dashboards.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Senior Infrastructure Engineer",
            experience_requirement="5 years of DevOps/infrastructure experience",
            original_description="Infrastructure role",
            required_skills=["Terraform", "Kubernetes"],
        )

        result = evaluator.evaluate(resume, jd)
        assert result["tenure_score"] == 100.0
        assert result["relevance_score"] == 0.0
        assert result["assigned_tier"] == ExperienceTier.UNRELATED_BACKGROUND
        assert result["dimension_score"] == 50.0

    def test_mixed_background_unrelated_ft_plus_relevant_internship(self, evaluator: ExperienceEvaluator):
        # Precedence 2: 3 years Retail Manager + 6 months Software Intern
        # Retail role provides 0.0 relevance; Software Intern earns Tier 3 (60%)
        # Retail tenure does NOT count in technical full-time tenure
        jobs = [
            WorkExperience(
                job_title="Retail Store Manager",
                company="BigStore",
                employment_start="2020",
                employment_end="2023",
                original_text="3 years managing store operations.",
            ),
            WorkExperience(
                job_title="Software Engineering Intern",
                company="TechLab",
                employment_type="Internship",
                technologies=["Python"],
                original_text="Internship building backend tools in Python.",
            ),
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title="Python Developer",
            experience_requirement="2 years experience required",
            original_description="Python dev role",
            required_skills=["Python"],
        )

        result = evaluator.evaluate(resume, jd)
        assert result["candidate_tenure_years"] == 0.0
        assert result["tenure_score"] == 0.0
        assert result["assigned_tier"] == ExperienceTier.RELEVANT_INTERNSHIP
        assert result["relevance_score"] == 60.0
        assert result["dimension_score"] == 30.0

    def test_applied_projects_precedence_tier_four(self, evaluator: ExperienceEvaluator):
        # Precedence 3: No employment, but verified relevant projects
        projects = [
            ProjectRecord(
                project_name="Distributed Pipeline",
                technologies=["Python", "FastAPI"],
                original_text="High performance backend pipeline in Python.",
            )
        ]
        resume = make_exp_resume(jobs=[], projects=projects)
        jd = JobDescriptionProfile(
            job_title="Backend Developer",
            experience_requirement="1 year experience",
            original_description="Backend role",
            required_skills=["Python", "FastAPI"],
        )

        result = evaluator.evaluate(resume, jd)
        assert result["candidate_tenure_years"] == 0.0
        assert result["assigned_tier"] == ExperienceTier.APPLIED_PROJECTS
        assert result["relevance_score"] == 40.0

    def test_scenario_three_neutralizes_relevance_when_jd_omits_target_role(self, evaluator: ExperienceEvaluator):
        # Scenario 3: JD requires 2 years experience, Python & SQL, but specifies NO target role/duties
        # Relevance is marked UNKNOWN_EVIDENCE and neutralized
        # Evaluated strictly on Tenure = 100%
        jobs = [
            WorkExperience(
                job_title="Developer",
                company="CodeInc",
                employment_start="2022",
                employment_end="2024",
                original_text="2 years experience.",
            )
        ]
        resume = make_exp_resume(jobs)
        jd = JobDescriptionProfile(
            job_title=None,
            experience_requirement="2 years minimum experience",
            original_description="Need someone with 2 years experience who knows Python and SQL.",
            required_skills=["Python", "SQL"],
        )

        result = evaluator.evaluate(resume, jd)
        assert result["tenure_score"] == 100.0
        assert result["relevance_score"] is None
        assert result["assigned_tier"] == ExperienceTier.UNKNOWN
        # Dimension score evaluated strictly on Tenure
        assert result["dimension_score"] == 100.0

