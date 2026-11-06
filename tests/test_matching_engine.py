"""
Integration and scenario tests for core/matching/engine.py.
Verifies end-to-end evaluation, anti-inflation redistribution, provisional gating,
fit classification boundary behavior, zero-criteria handling, and JSON serialization.
Validates exact numerical outcomes for Scenarios 1 through 5 from the approved Phase 4 specification.
"""
import pytest

from core.matching.engine import DeterministicMatchingEngine
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    JobDescriptionProfile,
    JobMatchResult,
    JobRequirement,
    RequirementCategory,
    RequirementPriority,
)
from models.resume_entities import (
    EducationRecord,
    ExtractedSkill,
    ProjectRecord,
    SkillCategory,
    WorkExperience,
)
from models.resume_profile import ResumeProfile


class TestDeterministicMatchingEngine:
    """Full integration and scenario testing suite for DeterministicMatchingEngine."""

    @pytest.fixture
    def engine(self) -> DeterministicMatchingEngine:
        return DeterministicMatchingEngine()

    def test_scenario_one_strong_candidate_match(self, engine: DeterministicMatchingEngine):
        """
        Scenario 1: Strong Candidate Match
        Skills (0.45): 87.5% -> 39.375
        Exp (0.25): 100.0% -> 25.000
        Edu (0.15): 100.0% -> 15.000
        Proj (0.15): 90.0% -> 13.500
        Total: 92.875% -> 92.9% (Exceptional Fit, Full Evaluation)
        """
        resume = ResumeProfile(
            filename="scenario1_resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python"),
                ExtractedSkill(name="PostgreSQL", original_mention="PostgreSQL"),
                ExtractedSkill(name="FastAPI", original_mention="FastAPI"),
                ExtractedSkill(name="Docker", original_mention="Docker"),
            ],
            experience=[
                WorkExperience(
                    job_title="Software Engineer",
                    company="CloudCorp",
                    employment_start="2020",
                    employment_end="2024",
                    employment_type="Full-time",
                    technologies=["Python", "FastAPI", "PostgreSQL"],
                    original_text="4 years backend development.",
                )
            ],
            education=[
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    institution="State University",
                    original_text="B.S. in Computer Science",
                )
            ],
            projects=[
                ProjectRecord(
                    project_name="Inventory Pipeline",
                    technologies=["Python", "FastAPI", "PostgreSQL"],
                    project_url="https://github.com/candidate/inventory",
                    description=(
                        "FastAPI order pipeline querying normalized PostgreSQL tables through an asynchronous connection pool, "
                        "reducing latency by 45% and handling 2.5k tps under peak loads."
                    ),
                    original_text="Project 1",
                ),
                ProjectRecord(
                    project_name="Tracking Service",
                    technologies=["Python", "FastAPI", "PostgreSQL"],
                    project_url="https://github.com/candidate/tracking",
                    description="Asynchronous service querying audit tables with sub-50ms latency.",
                    original_text="Project 2",
                ),
            ],
        )

        jd = JobDescriptionProfile(
            job_title="Backend Developer",
            company="Acme Cloud Solutions",
            original_description="Backend developer JD text",
            required_skills=["Python", "PostgreSQL", "FastAPI"],
            preferred_skills=["Docker", "AWS"],
            experience_requirement="3+ years of experience in backend development",
            education_requirement="Bachelor's degree required",
            responsibilities=[
                "Design and maintain high-performance asynchronous RESTful APIs.",
                "Optimize relational database schemas and queries for scale.",
            ],
            project_requirements=["Practical software projects expected."],
            requirement_counts={"total_requirements": 8, "technical_skills": 5},
            evaluation_eligibility=EvaluationStatus.FULL_EVALUATION,
        )

        result = engine.match(resume, jd)

        # Dimension scores
        sk_b = result.get_dimension("Technical Skills")
        exp_b = result.get_dimension("Work Experience")
        edu_b = result.get_dimension("Academic Education")
        proj_b = result.get_dimension("Engineering Projects")

        assert sk_b is not None and sk_b.dimension_score == 87.5
        assert exp_b is not None and exp_b.dimension_score == 100.0
        assert edu_b is not None and edu_b.dimension_score == 100.0
        assert proj_b is not None and proj_b.dimension_score == 90.0

        # Weights
        assert pytest.approx(sk_b.active_weight, rel=1e-3) == 0.45
        assert pytest.approx(exp_b.active_weight, rel=1e-3) == 0.25
        assert pytest.approx(edu_b.active_weight, rel=1e-3) == 0.15
        assert pytest.approx(proj_b.active_weight, rel=1e-3) == 0.15

        # Overall score & Fit
        assert pytest.approx(result.overall_score, rel=1e-3) == 92.875
        assert result.fit_classification == CandidateFitLevel.EXCEPTIONAL
        assert result.evaluation_status == EvaluationStatus.FULL_EVALUATION

    def test_scenario_two_partial_cloud_match(self, engine: DeterministicMatchingEngine):
        """
        Scenario 2: Partial Candidate Match (With Related Technology)
        Skills (0.45): 42.86% -> 19.286
        Exp (0.25): 68.75% -> 17.188
        Edu (0.15): 90.0% -> 13.500
        Proj (0.15): 15.29% -> 2.293
        Total: 52.267% -> 52.3% (Developing Fit, Full Evaluation)
        """
        resume = ResumeProfile(
            filename="scenario2_resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python"),
                ExtractedSkill(name="Flask", original_mention="Flask"),  # Related tech to FastAPI (factor 0.50)
            ],
            experience=[
                WorkExperience(
                    job_title="Junior Cloud Engineer",
                    company="CloudBase",
                    employment_start="2021",
                    employment_end="2023",
                    original_text="2.5 years of experience deploying basic cloud apps.",
                )
            ],
            education=[
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Information Systems",  # Adjacent STEM = 75%
                    institution="Tech College",
                    original_text="B.S. in Information Systems",
                )
            ],
            projects=[
                ProjectRecord(
                    project_name="Task App",
                    technologies=["Python", "Flask", "Docker"],
                    project_url=None,  # No URL -> B1 = 0
                    description="Simple web app to create and track personal tasks using Python, Flask, and Docker.",
                    original_text="Simple CRUD task tracker.",
                )
            ],
        )

        jd = JobDescriptionProfile(
            job_title="Senior Cloud Engineer",
            original_description="Cloud Engineer JD",
            required_skills=["Python", "FastAPI", "Kubernetes"],
            preferred_skills=["Redis"],
            experience_requirement="4+ years of cloud infrastructure experience",
            education_requirement="Bachelor's in Computer Science or related STEM field",
            responsibilities=["Design cloud infrastructure."],
            requirement_counts={"total_requirements": 7, "technical_skills": 4},
            evaluation_eligibility=EvaluationStatus.FULL_EVALUATION,
        )

        result = engine.match(resume, jd)

        sk_b = result.get_dimension("Technical Skills")
        exp_b = result.get_dimension("Work Experience")
        edu_b = result.get_dimension("Academic Education")
        proj_b = result.get_dimension("Engineering Projects")

        assert sk_b is not None and pytest.approx(sk_b.dimension_score, rel=1e-2) == 42.86
        assert exp_b is not None and pytest.approx(exp_b.dimension_score, rel=1e-2) == 68.75
        assert edu_b is not None and pytest.approx(edu_b.dimension_score, rel=1e-2) == 90.00
        assert proj_b is not None and pytest.approx(proj_b.dimension_score, rel=1e-2) == 15.29

        assert pytest.approx(result.overall_score, abs=0.1) == 52.3
        assert result.fit_classification == CandidateFitLevel.DEVELOPING
        assert result.evaluation_status == EvaluationStatus.FULL_EVALUATION

    def test_scenario_three_sparse_jd_provisional_match(self, engine: DeterministicMatchingEngine):
        """
        Scenario 3: Sparse JD with Insufficient Requirements
        JD: Python, SQL. 2 years tenure. No target role. No edu. No proj.
        Weights: Skills 0.60, Exp 0.40, Edu 0.00, Proj 0.00.
        Score: 100.0% -> Strictly displayed as PROVISIONAL.
        """
        resume = ResumeProfile(
            filename="scenario3_resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python"),
                ExtractedSkill(name="SQL", original_mention="SQL"),
            ],
            experience=[
                WorkExperience(
                    job_title="Developer",
                    company="CodeInc",
                    employment_start="2022",
                    employment_end="2024",
                    original_text="2 years experience.",
                )
            ],
        )

        jd = JobDescriptionProfile(
            job_title=None,
            original_description="We need someone with 2 years of experience who knows Python and SQL. Apply now!",
            required_skills=["Python", "SQL"],
            experience_requirement="2 years of experience",
            requirement_counts={"total_requirements": 3, "technical_skills": 2},
            evaluation_eligibility=EvaluationStatus.PROVISIONAL_EVALUATION,
        )

        result = engine.match(resume, jd)

        sk_b = result.get_dimension("Technical Skills")
        exp_b = result.get_dimension("Work Experience")
        edu_b = result.get_dimension("Academic Education")
        proj_b = result.get_dimension("Engineering Projects")

        # Capped weights
        assert sk_b is not None and sk_b.active_weight == 0.60
        assert exp_b is not None and exp_b.active_weight == 0.40
        assert edu_b is not None and edu_b.active_weight == 0.00 and edu_b.is_active is False
        assert proj_b is not None and proj_b.active_weight == 0.00 and proj_b.is_active is False

        assert sk_b.dimension_score == 100.0
        assert exp_b.dimension_score == 100.0  # Evaluated on tenure only

        assert result.overall_score == 100.0
        # Gating enforcement: PROVISIONAL overrides 100% numerical score
        assert result.evaluation_status == EvaluationStatus.PROVISIONAL_EVALUATION
        assert result.fit_classification == CandidateFitLevel.PROVISIONAL
        assert any("Sparse job description" in w for w in result.processing_warnings)

    def test_scenario_four_entry_level_fresh_graduate(self, engine: DeterministicMatchingEngine):
        """
        Scenario 4: Entry-Level Role with Fresh Graduate Candidate
        Skills (0.45): 83.33% -> 37.500
        Exp (0.25): 80.0% -> 20.000 (Tier 3 entry-level special, tenure neutralized)
        Edu (0.15): 100.0% -> 15.000
        Proj (0.15): 65.67% -> 9.850 (1 metric clause = 0 pts B2; no workflow = 0 pts B3)
        Total: 82.350% -> 82.35% (Strong Fit, Full Evaluation)
        """
        resume = ResumeProfile(
            filename="scenario4_resume.pdf",
            skills=[
                ExtractedSkill(name="Java", original_mention="Java"),
                ExtractedSkill(name="Spring Boot", original_mention="Spring Boot"),
                ExtractedSkill(name="PostgreSQL", original_mention="PostgreSQL"),  # Related tech to MySQL (0.50)
            ],
            experience=[
                WorkExperience(
                    job_title="Software Engineering Intern",
                    company="TechCorp",
                    employment_type="Internship",
                    technologies=["Java", "Spring Boot"],
                    original_text="Internship developing Java applications.",
                )
            ],
            education=[
                EducationRecord(
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    institution="University",
                    original_text="B.S. in Computer Science",
                )
            ],
            projects=[
                ProjectRecord(
                    project_name="University Portal",
                    technologies=["Java", "Spring Boot", "PostgreSQL", "Docker"],
                    project_url="https://github.com/candidate/portal",
                    description=(
                        "Implemented BCrypt auth and RBAC along with Hibernate query optimization reducing page load from 1.2s to 280ms."
                    ),
                    original_text="Project 1",
                ),
                ProjectRecord(
                    project_name="Book Exchange",
                    technologies=["Java", "Spring Boot"],
                    project_url=None,
                    description="Book exchange application in Java.",
                    original_text="Project 2",
                ),
            ],
        )

        jd = JobDescriptionProfile(
            job_title="Junior Software Engineer",
            original_description="Entry level Java developer",
            required_skills=["Java", "Spring Boot", "MySQL"],
            experience_requirement="Entry-Level (0 years required)",
            education_requirement="Bachelor's degree required",
            responsibilities=[
                "Develop and test modular web services.",
                "Write clean, maintainable, and well-documented code.",
            ],
            requirement_counts={"total_requirements": 6, "technical_skills": 3},
            evaluation_eligibility=EvaluationStatus.FULL_EVALUATION,
        )

        result = engine.match(resume, jd)

        sk_b = result.get_dimension("Technical Skills")
        exp_b = result.get_dimension("Work Experience")
        edu_b = result.get_dimension("Academic Education")
        proj_b = result.get_dimension("Engineering Projects")

        assert sk_b is not None and pytest.approx(sk_b.dimension_score, rel=1e-2) == 83.33
        assert exp_b is not None and exp_b.dimension_score == 80.0
        assert edu_b is not None and edu_b.dimension_score == 100.0
        assert proj_b is not None and pytest.approx(proj_b.dimension_score, rel=1e-2) == 65.67

        assert pytest.approx(result.overall_score, abs=0.1) == 82.35
        assert result.fit_classification == CandidateFitLevel.STRONG
        assert result.evaluation_status == EvaluationStatus.FULL_EVALUATION

    def test_scenario_five_complete_role_mismatch(self, engine: DeterministicMatchingEngine):
        """
        Scenario 5: Complete Role Mismatch with Unrelated Projects
        Skills (0.45): 0.0% -> 0.000
        Exp (0.25): 50.0% -> 12.500 (Tenure = 100%, Relevance = 0% Tier 5)
        Edu (0.15): 100.0% -> 15.000 (Generic Bachelor's satisfied by BA in Economics)
        Proj (0.15): 0.0% -> 0.000 (Unrelated marketing dashboards)
        Total: 27.500% -> 27.5% (Low Fit, Full Evaluation)
        """
        resume = ResumeProfile(
            filename="scenario5_resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python"),
                ExtractedSkill(name="Tableau", original_mention="Tableau"),
            ],
            experience=[
                WorkExperience(
                    job_title="Data Analyst",
                    company="RetailAnalytics",
                    employment_start="2019",
                    employment_end="2024",
                    original_text="5 years analyzing sales data in Tableau.",
                )
            ],
            education=[
                EducationRecord(
                    degree="Bachelor of Arts",
                    field_of_study="Economics",
                    institution="State College",
                    original_text="B.A. in Economics",
                )
            ],
            projects=[
                ProjectRecord(
                    project_name="Marketing Attribution Dashboard",
                    technologies=["Python", "Tableau"],
                    project_url="https://github.com/candidate/attribution",
                    description="Analyzed customer acquisition channels with marketing attribution models.",
                    original_text="Marketing project",
                )
            ],
        )

        jd = JobDescriptionProfile(
            job_title="Senior Infrastructure Engineer",
            original_description="Infrastructure engineer role",
            required_skills=["Go", "Kubernetes", "Terraform", "gRPC"],
            experience_requirement="5 years of infrastructure experience",
            education_requirement="Bachelor's degree required",
            responsibilities=["Build Terraform modules.", "Maintain Kubernetes clusters."],
            requirement_counts={"total_requirements": 8, "technical_skills": 4},
            evaluation_eligibility=EvaluationStatus.FULL_EVALUATION,
        )

        result = engine.match(resume, jd)

        sk_b = result.get_dimension("Technical Skills")
        exp_b = result.get_dimension("Work Experience")
        edu_b = result.get_dimension("Academic Education")
        proj_b = result.get_dimension("Engineering Projects")

        assert sk_b is not None and sk_b.dimension_score == 0.0
        assert exp_b is not None and exp_b.dimension_score == 50.0
        assert edu_b is not None and edu_b.dimension_score == 100.0
        assert proj_b is not None and proj_b.dimension_score == 0.0

        assert pytest.approx(result.overall_score, rel=1e-3) == 27.5
        assert result.fit_classification == CandidateFitLevel.LOW
        assert result.evaluation_status == EvaluationStatus.FULL_EVALUATION

    def test_zero_assessable_requirements_returns_insufficient_evidence(self, engine: DeterministicMatchingEngine):
        """When JD has 0 requirements, returns INSUFFICIENT_EVIDENCE with 100% reserve and 0% score."""
        resume = ResumeProfile(
            filename="resume.pdf",
            skills=[ExtractedSkill(name="Python", original_mention="Python")],
        )
        jd = JobDescriptionProfile(
            original_description="Join our team! Apply now.",
            required_skills=[],
            requirement_counts={"total_requirements": 0, "technical_skills": 0},
            evaluation_eligibility=EvaluationStatus.INSUFFICIENT_EVIDENCE,
        )

        result = engine.match(resume, jd)
        assert result.overall_score == 0.0
        assert result.evaluation_status == EvaluationStatus.INSUFFICIENT_EVIDENCE
        assert result.active_evaluation_coverage == 0.0
        assert result.unassessed_reserve == 100.0
        assert any("zero assessable requirements" in w for w in result.processing_warnings)

    def test_fit_classification_boundary_behavior(self, engine: DeterministicMatchingEngine):
        """Tests exact threshold boundary transitions."""
        # Check boundary mapping: >= 90: EXCEPTIONAL; 75-89.99: STRONG; 60-74.99: MODERATE; 40-59.99: DEVELOPING; < 40: LOW
        # Construct standard resume and test with varying scores
        resume = ResumeProfile(
            filename="resume.pdf",
            skills=[
                ExtractedSkill(name="Python", original_mention="Python"),
                ExtractedSkill(name="SQL", original_mention="SQL"),
                ExtractedSkill(name="FastAPI", original_mention="FastAPI"),
            ],
            experience=[
                WorkExperience(
                    job_title="Software Developer",
                    employment_start="2021",
                    employment_end="2024",
                    original_text="3 years",
                )
            ],
            education=[
                EducationRecord(degree="Bachelor of Science", original_text="B.S.")
            ],
        )
        jd = JobDescriptionProfile(
            job_title="Software Developer",
            original_description="Role description. Practical software projects expected.",
            required_skills=["Python", "SQL", "FastAPI"],
            experience_requirement="3 years experience",
            education_requirement="Bachelor's degree required",
            project_requirements=["Practical software projects expected."],
            requirement_counts={"total_requirements": 6, "technical_skills": 3},
        )
        res = engine.match(resume, jd)
        # Skills 100, Exp 100, Edu 100, Proj 0.
        # Overall = 0.45*100 + 0.25*100 + 0.15*100 + 0.15*0 = 85.0% -> STRONG
        assert res.overall_score == 85.0
        assert res.fit_classification == CandidateFitLevel.STRONG

    def test_json_serialization_round_trip(self, engine: DeterministicMatchingEngine):
        """JobMatchResult serializes and restores cleanly via Pydantic model_dump_json."""
        resume = ResumeProfile(
            filename="resume.pdf",
            skills=[ExtractedSkill(name="Python", original_mention="Python")],
        )
        jd = JobDescriptionProfile(
            original_description="Python role",
            required_skills=["Python", "FastAPI", "SQL"],
            requirement_counts={"total_requirements": 4, "technical_skills": 3},
        )
        result = engine.match(resume, jd)

        json_str = result.model_dump_json()
        restored = JobMatchResult.model_validate_json(json_str)

        assert restored.overall_score == result.overall_score
        assert restored.fit_classification == result.fit_classification
        assert restored.evaluation_status == result.evaluation_status
        assert len(restored.match_breakdown) == len(result.match_breakdown)
