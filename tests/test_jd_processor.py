"""
Unit and integration tests for Phase 4 Step 2: Job Description Processor.
Tests input validation, deterministic extraction, taxonomy canonicalization,
related-technology separation, deduplication, requirement counting,
provisional eligibility, StructuredOutputEngine integration, error handling,
secret redaction, and JSON serialization.
"""
import json
import pytest
from unittest.mock import MagicMock

from core.llm.exceptions import LLMTimeoutError
from core.llm.mock_provider import MockLLMProvider
from core.llm.models import LLMResponse, LLMUsageMetadata
from core.llm.structured.engine import StructuredOutputEngine
from models.matching import (
    EvaluationStatus,
    JobDescriptionProfile,
    RequirementCategory,
    RequirementPriority,
)
from processors.jd_processor import (
    MAX_JD_CHAR_LIMIT,
    JobDescriptionProcessor,
    process_job_description,
)


# =====================================================================
# Test Sample Job Descriptions
# =====================================================================

SAMPLE_BACKEND_JD = """
Job Title: Senior Backend Developer
Company: Acme Cloud Solutions

About the Role:
We are looking for a Senior Backend Developer to join our core distributed systems team.

Responsibilities:
• Design and maintain high-performance asynchronous RESTful APIs.
• Optimize relational database schemas and queries for scale.
• Collaborate with cross-functional teams on microservice architectures.

Requirements:
• 3+ years of experience in backend development.
• Strong proficiency in Python and FastAPI.
• Solid experience with PostgreSQL database design.
• Bachelor's degree in Computer Science or related STEM field.
• Demonstrated practical project experience or GitHub repository.

Preferred Qualifications:
• Experience with Docker and containerization.
• Knowledge of AWS cloud infrastructure.
• AWS Certified Solutions Architect is a plus.
"""

SAMPLE_SPARSE_JD = """
We need someone with 2 years of experience who knows Python and SQL. Apply now!
"""


# =====================================================================
# Test Group 1: Input Validation
# =====================================================================

class TestInputValidation:
    """Verifies input boundary handling, rejection of invalid inputs, and length checks."""

    def test_rejects_empty_string(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        with pytest.raises(ValueError) as exc_info:
            processor.process("")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_rejects_whitespace_only_string(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        with pytest.raises(ValueError) as exc_info:
            processor.process("   \n\t  ")
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_rejects_non_string_input(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        with pytest.raises(ValueError) as exc_info:
            processor.process(None)  # type: ignore
        assert "must be a string" in str(exc_info.value)

    def test_truncates_excessively_long_input(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        excessive_text = "Python developer. " * 2500  # ~45,000 characters > 30,000 limit
        profile = processor.process(excessive_text)
        assert len(profile.original_description) == MAX_JD_CHAR_LIMIT
        assert any("truncated to 30,000 characters" in w for w in profile.processing_warnings)

    def test_short_valid_jd_processed_successfully(self):
        """Short JD with sufficient assessable requirements achieves FULL_EVALUATION."""
        short_valid_jd = "Python, PostgreSQL, AWS, and Docker developer: 3 years experience."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(short_valid_jd)

        # Requirements extracted without hallucinating
        assert profile.job_title is not None
        assert profile.experience_requirement is not None
        assert set(profile.required_skills) == {"Python", "PostgreSQL", "AWS", "Docker"}

        # Counts: 4 technical skills (>= 3) and >= 4 total requirements
        assert profile.requirement_counts["technical_skills"] == 4
        assert profile.requirement_counts["total_requirements"] >= 4
        assert profile.evaluation_eligibility == EvaluationStatus.FULL_EVALUATION
        assert profile.is_provisional is False

    def test_short_sparse_jd_triggers_provisional_evaluation(self):
        """Short JD below threshold triggers PROVISIONAL_EVALUATION using counting rules."""
        short_sparse_jd = "Python and SQL developer: 2 years experience."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(short_sparse_jd)

        # Technical skills = 2 (< 3 threshold) triggers provisional evaluation
        assert profile.requirement_counts["technical_skills"] == 2
        assert profile.evaluation_eligibility == EvaluationStatus.PROVISIONAL_EVALUATION
        assert profile.is_provisional is True
        assert any("Sparse job description" in w for w in profile.processing_warnings)

    def test_short_text_without_meaningful_requirements(self):
        """Short text with zero assessable requirements returns clear insufficient-evidence warning without fabricating."""
        short_unrelated_text = "Join our friendly local team! Great pay, flexible hours, apply today."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(short_unrelated_text)

        # Zero requirements fabricated
        assert len(profile.normalized_requirements) == 0
        assert len(profile.required_skills) == 0
        assert len(profile.preferred_skills) == 0
        assert profile.experience_requirement is None
        assert profile.education_requirement is None
        assert profile.requirement_counts["total_requirements"] == 0
        assert profile.requirement_counts["technical_skills"] == 0

        # Clear insufficient evidence warning and provisional status
        assert profile.evaluation_eligibility == EvaluationStatus.PROVISIONAL_EVALUATION
        assert profile.is_provisional is True
        assert any("Insufficient evidence for job matching" in w for w in profile.processing_warnings)

    def test_warns_on_extremely_short_description(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        short_jd = "Looking for a Python developer immediately."
        profile = processor.process(short_jd)
        assert any("extremely short" in w.lower() for w in profile.processing_warnings)


# =====================================================================
# Test Group 2: Deterministic Extraction and Canonicalization
# =====================================================================

class TestDeterministicExtraction:
    """Tests deterministic parsing of titles, companies, skills, requirements, and taxonomy canonicalization."""

    def test_valid_technical_jd_extraction(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(SAMPLE_BACKEND_JD)

        # Metadata
        assert profile.job_title == "Senior Backend Developer"
        assert profile.company == "Acme Cloud Solutions"
        assert "3+ years of experience" in (profile.experience_requirement or "")
        assert "Bachelor" in (profile.education_requirement or "")

        # Required skills
        assert "Python" in profile.required_skills
        assert "FastAPI" in profile.required_skills
        assert "PostgreSQL" in profile.required_skills

        # Preferred skills
        assert "Docker" in profile.preferred_skills
        assert "AWS" in profile.preferred_skills

        # Responsibilities
        assert len(profile.responsibilities) >= 2
        assert any("asynchronous RESTful APIs" in r for r in profile.responsibilities)

        # Certifications and Projects
        assert any("AWS Certified" in c for c in profile.certifications)
        assert len(profile.project_requirements) >= 1

    def test_missing_title_handled_gracefully(self):
        jd_no_title = """
        Company: CloudTech
        We need an experienced developer who knows Go and Docker with 4 years experience.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_no_title)
        # Does not hallucinate or invent title
        assert profile.company == "CloudTech"
        assert "Go" in profile.required_skills

    def test_missing_company_handled_gracefully(self):
        jd_no_company = """
        Job Title: Infrastructure Architect
        Requirements: 5 years experience with Kubernetes and Terraform.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_no_company)
        assert profile.job_title == "Infrastructure Architect"
        assert profile.company is None
        assert "Kubernetes" in profile.required_skills

    def test_user_supplied_overrides_take_precedence(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(
            SAMPLE_BACKEND_JD,
            job_title_override="Lead Python Architect",
            company_override="Override Global Inc",
        )
        assert profile.job_title == "Lead Python Architect"
        assert profile.company == "Override Global Inc"

    def test_synonym_canonicalization(self):
        jd_synonyms = """
        Job Title: Cloud Engineer
        Requirements:
        • Experience with K8s and Postgres.
        • Proficient in JS and TS.
        • Experience with py and golang.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_synonyms)

        # Synonyms mapped to canonical names
        assert "Kubernetes" in profile.required_skills
        assert "PostgreSQL" in profile.required_skills
        assert "JavaScript" in profile.required_skills
        assert "TypeScript" in profile.required_skills
        assert "Python" in profile.required_skills
        assert "Go" in profile.required_skills

    def test_duplicate_skills_deduplicated_with_priority_precedence(self):
        jd_duplicate_skills = """
        Job Title: Software Engineer
        Requirements:
        • Python and PostgreSQL.
        Responsibilities:
        • Write clean Python code.
        Preferred:
        • Python, Docker, and PostgreSQL.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_duplicate_skills)

        # Python and PostgreSQL must each appear exactly ONCE
        assert profile.required_skills.count("Python") == 1
        assert profile.required_skills.count("PostgreSQL") == 1
        # Required takes priority: must NOT be in preferred
        assert "Python" not in profile.preferred_skills
        assert "PostgreSQL" not in profile.preferred_skills
        assert "Docker" in profile.preferred_skills

    def test_related_technologies_remain_distinct(self):
        """FastAPI and Flask are related technologies, but must remain distinct requirements."""
        jd_related = """
        Job Title: Python Web Developer
        Requirements:
        • Experience with FastAPI and Flask microframeworks.
        • Familiarity with React and Angular.
        • Experience with PostgreSQL and MySQL.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_related)

        assert "FastAPI" in profile.required_skills
        assert "Flask" in profile.required_skills
        assert "React" in profile.required_skills
        assert "Angular" in profile.required_skills
        assert "PostgreSQL" in profile.required_skills
        assert "MySQL" in profile.required_skills

        # Distinct requirement items
        req_names = [r.canonical_name for r in profile.normalized_requirements]
        assert "FastAPI" in req_names
        assert "Flask" in req_names
        assert "React" in req_names
        assert "Angular" in req_names


# =====================================================================
# Test Group 3: Education and Experience Requirements
# =====================================================================

class TestEducationAndExperienceExtraction:
    """Tests generic vs specific degree extraction and tenure parsing."""

    def test_generic_degree_requirement_extraction(self):
        jd = "Job Title: Developer\nRequirements: Bachelor's degree required. 2+ years of experience. Python, SQL, Git."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd)

        assert "Bachelor" in (profile.education_requirement or "")
        # Does not invent a specific major
        assert "Computer Science" not in (profile.education_requirement or "")

    def test_discipline_specific_degree_requirement_extraction(self):
        jd = "Job Title: Research Scientist\nRequirements: Ph.D. in Computer Science or Electrical Engineering required. Python, PyTorch, C++."
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd)

        assert "Ph.D." in (profile.education_requirement or "")
        assert "Computer Science" in (profile.education_requirement or "")

    def test_repeated_responsibilities_deduplicated(self):
        jd = """
        Job Title: Lead Engineer
        Responsibilities:
        • Build scalable distributed systems.
        • Build scalable distributed systems.
        • Mentor junior engineering staff.
        Requirements: Python, Go, Docker.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd)
        assert len(profile.responsibilities) == 2
        assert profile.responsibilities.count("Build scalable distributed systems.") == 1


# =====================================================================
# Test Group 4: Requirement Counting and Provisional Eligibility
# =====================================================================

class TestRequirementCountingAndProvisionalEligibility:
    """Verifies the exact requirement-counting policy and provisional assessment threshold."""

    def test_full_evaluation_threshold_met(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(SAMPLE_BACKEND_JD)

        counts = profile.requirement_counts
        assert counts["required_technical_skills"] >= 3
        assert counts["preferred_technical_skills"] >= 1
        assert counts["technical_skills"] >= 4
        assert counts["total_requirements"] >= 4

        assert profile.evaluation_eligibility == EvaluationStatus.FULL_EVALUATION
        assert profile.is_provisional is False

    def test_sparse_jd_triggers_provisional_eligibility(self):
        """Sparse JD with only 2 skills and 3 total reqs fails the threshold."""
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(SAMPLE_SPARSE_JD)

        counts = profile.requirement_counts
        assert counts["technical_skills"] == 2  # < 3 threshold
        assert counts["total_requirements"] == 3  # < 4 threshold

        assert profile.evaluation_eligibility == EvaluationStatus.PROVISIONAL_EVALUATION
        assert profile.is_provisional is True
        assert any("Sparse job description" in w for w in profile.processing_warnings)

    def test_preferred_skills_count_toward_technical_skills(self):
        jd_with_pref = """
        Job Title: Developer
        Requirements: Python, PostgreSQL.
        Preferred: Docker.
        """
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(jd_with_pref)

        counts = profile.requirement_counts
        assert counts["required_technical_skills"] == 2
        assert counts["preferred_technical_skills"] == 1
        # Total skills = 2 req + 1 pref = 3 skills (meets skill threshold!)
        assert counts["technical_skills"] == 3


# =====================================================================
# Test Group 5: StructuredOutputEngine & MockLLMProvider Integration
# =====================================================================

class TestStructuredEngineIntegration:
    """Tests LLM semantic extraction, error fallback, secret scrubbing, and mock provider operation."""

    def test_successful_llm_extraction_via_mock_provider(self):
        mock_canned_json = json.dumps({
            "job_title": "AI Platform Engineer",
            "company": "DeepAI Labs",
            "experience_requirement": "4+ years",
            "education_requirement": "Master's in Computer Science",
            "required_skills": ["Python", "PyTorch", "Docker"],
            "preferred_skills": ["Kubernetes", "FastAPI"],
            "responsibilities": ["Deploy large language model inference clusters", "Optimize GPU throughput"],
            "certifications": ["NVIDIA Certified Associate"],
            "project_requirements": ["Open-source contributions to machine learning repositories"],
        })

        mock_prov = MockLLMProvider(canned_response=mock_canned_json)
        engine = StructuredOutputEngine(dispatcher=mock_prov)
        processor = JobDescriptionProcessor(structured_engine=engine, enable_llm_extraction=True)

        profile = processor.process(
            "We are DeepAI Labs looking for an AI Platform Engineer with PyTorch experience."
        )

        assert profile.job_title == "AI Platform Engineer"
        assert profile.company == "DeepAI Labs"
        assert "Python" in profile.required_skills
        assert "PyTorch" in profile.required_skills
        assert "Kubernetes" in profile.preferred_skills
        assert profile.evaluation_eligibility == EvaluationStatus.FULL_EVALUATION

    def test_llm_failure_triggers_deterministic_fallback(self):
        """If the LLM raises a timeout or parsing error, processor falls back to deterministic extraction."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = LLMTimeoutError("Groq timeout after 30s", provider_name="groq")

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        processor = JobDescriptionProcessor(structured_engine=engine, enable_llm_extraction=True)

        profile = processor.process(SAMPLE_BACKEND_JD)

        # Fallback successfully extracted data
        assert profile.job_title == "Senior Backend Developer"
        assert "Python" in profile.required_skills
        # Warning recorded indicating fallback
        assert any("LLM semantic extraction failed" in w for w in profile.processing_warnings)

    def test_secret_scrubbing_in_processing_warnings(self):
        """API keys inside exceptions are scrubbed before being recorded in processing warnings."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.side_effect = Exception("Auth failed for key gsk_SecretKey12345678901234567890")

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        processor = JobDescriptionProcessor(structured_engine=engine, enable_llm_extraction=True)

        profile = processor.process(SAMPLE_BACKEND_JD)
        all_warnings = " ".join(profile.processing_warnings)

        assert "gsk_" not in all_warnings
        assert "[REDACTED_API_KEY]" in all_warnings

    def test_privacy_flags_on_llm_request(self):
        """Verifies JD extraction requests preserve expected privacy settings."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.generate.return_value = LLMResponse(
            content=json.dumps({"job_title": "Dev", "required_skills": ["Python"]}),
            provider_name="mock",
            model_name="mock-model",
        )

        engine = StructuredOutputEngine(dispatcher=mock_dispatcher)
        processor = JobDescriptionProcessor(structured_engine=engine, enable_llm_extraction=True)
        processor.process("Job Title: Dev\nRequirements: Python")

        first_call_req = mock_dispatcher.generate.call_args_list[0][0][0]
        # JDs are public postings: is_sensitive=False
        assert first_call_req.is_sensitive is False
        assert first_call_req.allow_fallback is True


# =====================================================================
# Test Group 6: JSON Serialization & Round Trip
# =====================================================================

class TestJsonSerialization:
    """Verifies that processed profiles serialize and deserialize without loss."""

    def test_job_description_profile_json_round_trip(self):
        processor = JobDescriptionProcessor(enable_llm_extraction=False)
        profile = processor.process(SAMPLE_BACKEND_JD)

        json_str = profile.model_dump_json()
        restored = JobDescriptionProfile.model_validate_json(json_str)

        assert restored.job_title == profile.job_title
        assert restored.company == profile.company
        assert len(restored.normalized_requirements) == len(profile.normalized_requirements)
        assert restored.requirement_counts == profile.requirement_counts
        assert restored.evaluation_eligibility == profile.evaluation_eligibility

    def test_functional_wrapper(self):
        profile = process_job_description("Job Title: Go Developer\nRequirements: Go, Docker, AWS.")
        assert profile.job_title == "Go Developer"
        assert "Go" in profile.required_skills

