"""
Unit tests for core/matching/project_evaluator.py.
Verifies the three-pillar project scoring formula (C_tech + D_context + V_align),
artifact URL validation, quantified metric thresholding, toolchain workflow requirements,
unrelated project filtering, and senior engineer neutralization.
"""
import pytest

from core.matching.project_evaluator import (
    ProjectEvaluator,
    validate_artifact_url,
)
from models.matching import (
    EvidenceStatus,
    EvidenceVerificationLevel,
    JobDescriptionProfile,
)
from models.resume_entities import ProjectRecord
from models.resume_profile import ResumeProfile


def make_proj_resume(projects: list[ProjectRecord]) -> ResumeProfile:
    return ResumeProfile(
        filename="proj_resume.pdf",
        projects=projects,
    )


class TestProjectEvaluator:
    """Tests deterministic project scoring across all three pillars and edge cases."""

    @pytest.fixture
    def evaluator(self) -> ProjectEvaluator:
        return ProjectEvaluator()

    def test_scenario_one_project_scoring(self, evaluator: ProjectEvaluator):
        # Scenario 1:
        # Req: Python, PostgreSQL, FastAPI. Pref: Docker, AWS.
        # Candidate has 2 projects implementing Python, FastAPI, Postgres.
        # Docker and AWS absent from projects.
        # C_tech = 40.0 * (1.0 + 1.0 + 1.0) / (3.0 + 0.5 + 0.5) = 40.0 * (3.0 / 4.0) = 30.00
        # D_context = 10.0 (URLs on both) + 15.0 (>= 2 metrics clauses) + 10.0 (toolchain workflow) = 35.00
        # V_align = 25.00 (High alignment with core backend duties)
        # S_proj = 30.00 + 35.00 + 25.00 = 90.00%
        p1 = ProjectRecord(
            project_name="Asynchronous Inventory Pipeline",
            technologies=["Python", "FastAPI", "PostgreSQL"],
            project_url="https://github.com/candidate/inventory-service",
            description=(
                "FastAPI order pipeline querying normalized PostgreSQL tables through an asynchronous connection pool, "
                "reducing latency by 45% and handling 2.5k tps under peak loads."
            ),
            original_text="Project 1 description",
        )
        p2 = ProjectRecord(
            project_name="Audit Tracking API",
            technologies=["Python", "FastAPI", "PostgreSQL"],
            project_url="https://github.com/candidate/audit-service",
            description="RESTful service querying audit records with sub-50ms latency.",
            original_text="Project 2 description",
        )
        resume = make_proj_resume([p1, p2])
        jd = JobDescriptionProfile(
            original_description="Backend developer role",
            required_skills=["Python", "PostgreSQL", "FastAPI"],
            preferred_skills=["Docker", "AWS"],
            responsibilities=[
                "Design and maintain high-performance asynchronous RESTful APIs.",
                "Optimize relational database schemas and queries for scale.",
            ],
            project_requirements=["Demonstrated practical project experience on GitHub."],
        )

        result = evaluator.evaluate(resume, jd, candidate_ft_years=4.0)
        assert pytest.approx(result["c_tech"], rel=1e-3) == 30.0
        assert pytest.approx(result["d_context"], rel=1e-3) == 35.0
        assert pytest.approx(result["v_align"], rel=1e-3) == 25.0
        assert pytest.approx(result["dimension_score"], rel=1e-3) == 90.0
        assert result["is_active"] is True

    def test_single_quantified_clause_earns_zero_metric_points(self, evaluator: ProjectEvaluator):
        # Scenario 4 refinement: Rubric strictly requires >= 2 distinct quantified action clauses.
        # Exactly 1 quantified metric clause earns 0.0 points in Pillar B2.
        p1 = ProjectRecord(
            project_name="Campus Management Portal",
            technologies=["Java", "Spring Boot", "PostgreSQL"],
            project_url="https://github.com/candidate/campus-portal",
            description="Implemented BCrypt auth and Hibernate query optimization reducing page load from 1.2s to 280ms.",
            original_text="Project description",
        )
        resume = make_proj_resume([p1])
        jd = JobDescriptionProfile(
            original_description="Java Developer role",
            required_skills=["Java", "Spring Boot"],
            responsibilities=["Develop web applications."],
        )

        result = evaluator.evaluate(resume, jd)
        # B1: URL present = 10.0
        # B2: Only 1 metric clause ("from 1.2s to 280ms") -> 0.0 points
        # B3: 3 tools listed but no operational workflow -> 0.0 points
        # D_context = 10.0 + 0.0 + 0.0 = 10.0
        assert result["d_context"] == 10.0

    def test_shallow_tool_list_without_workflow_earns_zero_toolchain_points(self, evaluator: ProjectEvaluator):
        # Scenario 2 refinement: Merely listing 3+ tools without demonstrating interaction workflow earns 0.0 points.
        p1 = ProjectRecord(
            project_name="Personal Task App",
            technologies=["Python", "Flask", "Docker"],
            description="Simple web app to create and track personal tasks using Python, Flask, and Docker.",
            original_text="Simple CRUD task tracker.",
        )
        resume = make_proj_resume([p1])
        jd = JobDescriptionProfile(
            original_description="Cloud role",
            required_skills=["Python", "FastAPI"],
            responsibilities=["Deploy cloud apps."],
        )

        result = evaluator.evaluate(resume, jd)
        # URL = 0, Metrics = 0, Toolchain = 0 -> D_context = 0.0
        assert result["d_context"] == 0.0

    def test_unrelated_projects_receive_zero_compatibility_points(self, evaluator: ProjectEvaluator):
        # Scenario 5: Data Analyst projects (Marketing dashboards in Python & Tableau)
        # applying for Senior Infrastructure Engineer (Terraform, Kubernetes, Go)
        p1 = ProjectRecord(
            project_name="Marketing Attribution Dashboard",
            technologies=["Python", "Tableau"],
            project_url="https://github.com/candidate/attribution",
            description="Created marketing attribution reports and visualizations in Tableau.",
            original_text="Marketing dashboard project.",
        )
        resume = make_proj_resume([p1])
        jd = JobDescriptionProfile(
            original_description="Senior Infrastructure Engineer",
            required_skills=["Terraform", "Kubernetes", "Go", "gRPC"],
            responsibilities=["Design CI/CD pipelines and infrastructure as code."],
        )

        result = evaluator.evaluate(resume, jd)
        # Zero infrastructure overlap -> 0.0 points across all pillars
        assert result["c_tech"] == 0.0
        assert result["d_context"] == 0.0
        assert result["v_align"] == 0.0
        assert result["dimension_score"] == 0.0
        assert result["evidence_status"] == EvidenceStatus.MISSING_EVIDENCE

    def test_seniority_does_not_independently_neutralize_project_scoring(self, evaluator: ProjectEvaluator):
        # A senior candidate with 8 years of tenure who provides relevant projects
        # MUST have their projects evaluated by the rubric, NOT neutralized.
        p = ProjectRecord(
            project_name="Distributed Cache",
            technologies=["Python", "PostgreSQL"],
            project_url="https://github.com/candidate/cache",
            description="Built distributed cache querying database through pool. Optimized page load from 1.5s to 300ms, reduced latency by 40%.",
            original_text="Text",
        )
        resume = make_proj_resume([p])
        jd = JobDescriptionProfile(
            original_description="Senior Staff Backend Engineer. 7+ years experience.",
            required_skills=["Python", "PostgreSQL"],
            project_requirements=[],
        )

        result = evaluator.evaluate(resume, jd, candidate_ft_years=8.0)
        assert result["is_active"] is True
        assert result["dimension_score"] > 0.0
        assert result["evidence_status"] in (EvidenceStatus.VERIFIED_EVIDENCE, EvidenceStatus.PARTIAL_EVIDENCE)

    def test_no_project_requirement_and_no_candidate_projects_neutralizes_dimension(self, evaluator: ProjectEvaluator):
        # When JD has no project requirements AND candidate has no projects,
        # the Projects dimension is neutralized regardless of whether candidate has 0 or 10 years experience.
        resume = make_proj_resume([])
        jd = JobDescriptionProfile(
            original_description="Software Engineer. Core development.",
            required_skills=["Python"],
            project_requirements=[],
        )

        # Junior candidate (1 year experience)
        res_jr = evaluator.evaluate(resume, jd, candidate_ft_years=1.0)
        assert res_jr["is_active"] is False
        assert res_jr["dimension_score"] == 0.0
        assert res_jr["evidence_status"] == EvidenceStatus.UNKNOWN_EVIDENCE

        # Senior candidate (10 years experience)
        res_sr = evaluator.evaluate(resume, jd, candidate_ft_years=10.0)
        assert res_sr["is_active"] is False
        assert res_sr["dimension_score"] == 0.0
        assert res_sr["evidence_status"] == EvidenceStatus.UNKNOWN_EVIDENCE

    def test_valid_https_github_gitlab_bitbucket_urls(self, evaluator: ProjectEvaluator):
        # Structurally valid HTTPS URLs to recognized artifact hosts
        for host_url in [
            "https://github.com/candidate/portfolio-api",
            "https://gitlab.com/candidate/microservice",
            "https://bitbucket.org/candidate/core-engine",
            "https://candidate.github.io/demo-app",
        ]:
            is_valid, level, reason = validate_artifact_url(host_url)
            assert is_valid is True
            assert level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL

    def test_invalid_url_strings_and_unsupported_schemes(self, evaluator: ProjectEvaluator):
        # Invalid schemes, arbitrary unaccepted domains, and malformed strings
        cases = [
            ("http://github.com/candidate/repo", EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL),
            ("ftp://github.com/candidate/repo", EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL),
            ("https://randomdomain.example.com/repo", EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL),
            ("https://github.com", EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL),  # empty path
            ("not-a-valid-url", EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL),
            ("", EvidenceVerificationLevel.SELF_REPORTED),
            ("   ", EvidenceVerificationLevel.SELF_REPORTED),
            (None, EvidenceVerificationLevel.SELF_REPORTED),
        ]
        for url, expected_level in cases:
            is_valid, level, reason = validate_artifact_url(url)
            assert is_valid is False
            assert level == expected_level

    def test_structurally_valid_urls_not_represented_as_inspected_repositories(self, evaluator: ProjectEvaluator):
        # Structurally valid URL awards 10 points for Pillar B1, but must NOT be labeled
        # as ARTIFACT_SUPPORTED (which is reserved for inspected content) or INDEPENDENTLY_VERIFIED.
        p = ProjectRecord(
            project_name="Auth Service",
            technologies=["Python"],
            project_url="https://github.com/candidate/auth-service",
            description="Service built with Python.",
            original_text="Text",
        )
        resume = make_proj_resume([p])
        jd = JobDescriptionProfile(
            original_description="Python dev",
            required_skills=["Python"],
        )

        result = evaluator.evaluate(resume, jd)
        records = result["project_evidence"]
        assert len(records) == 1
        assert records[0].evidence_verification_level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL
        assert records[0].evidence_verification_level != EvidenceVerificationLevel.ARTIFACT_SUPPORTED
        assert records[0].evidence_verification_level != EvidenceVerificationLevel.INDEPENDENTLY_VERIFIED

        # Check diagnostics note offline validation
        diag_text = " ".join(result["diagnostics"])
        assert "offline evaluation" in diag_text.lower() or "not inspected" in diag_text.lower()
        assert result["d_context"] == 10.0  # 10 points for B1

    def test_zero_network_calls_during_url_validation(self, evaluator: ProjectEvaluator, monkeypatch: pytest.MonkeyPatch):
        # Ensure url validation is strictly offline: blocking socket creation must not raise
        import socket

        def forbid_socket(*args, **kwargs):
            raise AssertionError("Network socket called during offline URL validation!")

        monkeypatch.setattr(socket, "socket", forbid_socket)

        # Run URL validation and evaluation with forbidden socket
        is_valid, level, _ = validate_artifact_url("https://github.com/candidate/test-repo")
        assert is_valid is True
        assert level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL

    def test_url_credentials_and_port_validation(self, evaluator: ProjectEvaluator):
        # 1. Embedded credentials must be rejected
        cred_url = "https://user:password@github.com/user/repository"
        is_valid, level, reason = validate_artifact_url(cred_url)
        assert is_valid is False
        assert level == EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL
        assert "credentials" in reason.lower()

        # 2. Unexpected ports must be rejected
        for bad_port_url in [
            "https://github.com:8443/user/repository",
            "https://github.com:444/user/repository",
            "https://github.com:8080/user/repository",
            "https://gitlab.com:80/user/repository",
        ]:
            is_valid, level, reason = validate_artifact_url(bad_port_url)
            assert is_valid is False
            assert level == EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL
            assert "unexpected port" in reason.lower()

        # 3. Default HTTPS port (443) and standard portless URLs are accepted
        for valid_url in [
            "https://github.com:443/user/repository",
            "https://github.com/user/repository",
            "https://gitlab.com:443/candidate/repo",
        ]:
            is_valid, level, reason = validate_artifact_url(valid_url)
            assert is_valid is True
            assert level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL

        # 4. Malformed port values must be handled safely without unhandled exceptions escaping
        for malformed_port_url in [
            "https://github.com:abc/user/repository",
            "https://github.com:999999/user/repository",
            "https://github.com:-1/user/repository",
            "https://github.com:/user/repository",
        ]:
            is_valid, level, reason = validate_artifact_url(malformed_port_url)
            assert is_valid is False
            assert level == EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL

        # 5. Verify in ProjectEvaluator scoring that credential/bad-port URLs receive 0 Pillar B1 points
        bad_proj = ProjectRecord(
            project_name="Creds Project",
            technologies=["Python"],
            project_url="https://user:password@github.com/user/repository",
            description="Project with credentials in URL.",
            original_text="Text",
        )
        port_proj = ProjectRecord(
            project_name="Port Project",
            technologies=["Python"],
            project_url="https://github.com:8443/user/repository",
            description="Project with unexpected port in URL.",
            original_text="Text",
        )
        good_proj = ProjectRecord(
            project_name="Valid Port Project",
            technologies=["Python"],
            project_url="https://github.com:443/user/repository",
            description="Project with standard 443 port.",
            original_text="Text",
        )

        resume_bad = make_proj_resume([bad_proj, port_proj])
        jd = JobDescriptionProfile(original_description="Python dev", required_skills=["Python"])
        res_bad = evaluator.evaluate(resume_bad, jd)
        assert res_bad["d_context"] == 0.0  # Zero points for Pillar B1
        assert all(rec.evidence_verification_level == EvidenceVerificationLevel.CANDIDATE_PROVIDED_URL for rec in res_bad["project_evidence"])

        resume_good = make_proj_resume([good_proj])
        res_good = evaluator.evaluate(resume_good, jd)
        assert res_good["d_context"] == 10.0  # 10 points for valid port 443
        assert res_good["project_evidence"][0].evidence_verification_level == EvidenceVerificationLevel.STRUCTURALLY_VALID_URL

