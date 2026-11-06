"""
End-to-end integration tests for CareerMatchingPipeline (Phase 4 Step 5).
Validates dual input contracts, deterministic scoring immutability,
allowlist JSON export privacy, rule-based fallback on LLM failure,
status callbacks, repeated analyses isolation, and socket isolation.
"""
import io
import json
import pytest
import pymupdf

from core.llm.exceptions import LLMRateLimitError
from core.llm.mock_provider import MockLLMProvider
from core.llm.structured.engine import StructuredOutputEngine
from core.matching.engine import DeterministicMatchingEngine
from core.matching.exceptions import PipelineInputError
from core.matching.explanation_generator import MatchExplainer
from core.matching.pipeline import CareerMatchingPipeline, PipelineMatchResult
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    EvidenceStatus,
    JobDescriptionProfile,
    JobMatchResult,
    MatchExplanation,
)
from models.resume_profile import ProfileProcessingStatus, ResumeProfile
from processors.jd_processor import JobDescriptionProcessor
from processors.pdf_processor import PDFProcessingError
from processors.resume_pipeline import ResumePipeline


# ---------------------------------------------------------------------------
# Synthetic Test Document Helpers
# ---------------------------------------------------------------------------

def make_test_pdf(text: str) -> bytes:
    """Generate in-memory PDF bytes from text using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=9)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


SAMPLE_RESUME_TEXT = """
Jane Doe
jane.doe@example.com | (555) 987-6543 | San Francisco, CA

PROFESSIONAL SUMMARY
Senior Backend Engineer with 5 years of experience building distributed systems in Python and Go.

TECHNICAL SKILLS
Languages: Python, Go, SQL
Frameworks & Databases: FastAPI, PostgreSQL, Redis, Docker
Cloud: AWS

WORK EXPERIENCE
Senior Backend Engineer at CloudScale Technologies | 2021 - Present
- Architected and scaled high-throughput FastAPI microservices serving 10M daily requests.
- Designed distributed PostgreSQL connection pooling, reducing query latencies by 35%.

Software Engineer at DataStream Inc | 2019 - 2021
- Developed REST APIs in Python and integrated Redis caching layers.

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2015 - 2019

PROJECTS
Distributed Task Queue | https://github.com/janedoe/task-queue
- High-performance distributed task scheduler in Python and Redis.
- Reduced queue backlog latency from 2s to 150ms under peak load.
"""

SAMPLE_JD_FULL = """
Job Title: Senior Backend Engineer
Company: Stripe

About the Role:
We are seeking an experienced Senior Backend Engineer to join our core platform engineering team.

Requirements:
- 4+ years of professional backend software engineering experience.
- Strong proficiency in Python and modern frameworks like FastAPI.
- Deep expertise in relational databases including PostgreSQL.
- Experience with distributed caching using Redis.
- Bachelor's degree in Computer Science, Computer Engineering, or related STEM field.

Preferred Qualifications:
- Experience with AWS cloud infrastructure and Docker containers.
- Familiarity with Go or high-concurrency systems.
"""

SAMPLE_JD_SPARSE = """
Job Title: Python Developer
Company: QuickStart

Looking for a developer.
Must know Python and Docker.
"""

SAMPLE_JD_ZERO_REQUIREMENTS = """
Company: Vague Corp
We are a dynamic company looking for rockstars who can wear many hats and synergize paradigms.
"""

SAMPLE_EXPLANATION_JSON = json.dumps({
    "overall_match_summary": "Strong match with verified backend engineering proficiency in Python, FastAPI, and PostgreSQL.",
    "key_strengths": ["Demonstrated 5 years of backend engineering experience", "Core proficiency in Python, FastAPI, and PostgreSQL"],
    "skill_gaps_assessment": ["AWS cloud deployment not demonstrated in the supplied resume"],
    "experience_assessment": "Exceeds the 4-year tenure requirement with 5 years of direct full-time backend engineering.",
    "education_assessment": "Holds a Bachelor of Science in Computer Science meeting core STEM qualification requirements.",
    "project_assessment": "Engineering project demonstrates hands-on distributed queuing with Python and Redis; offline link validated.",
    "improvement_recommendations": ["Expand exposure to containerized AWS deployments."],
    "interview_preparation_suggestions": ["Be prepared to discuss distributed database indexing and high-throughput locking strategies."],
    "evidence_limitations_and_warnings": ["Candidate repository contents were not inspected offline; metrics are self-reported candidate claims."],
})


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestCareerMatchingPipeline:
    """Comprehensive integration tests for CareerMatchingPipeline."""

    def _build_pipeline(self, mock_response: str = SAMPLE_EXPLANATION_JSON, simulate_error: Exception | None = None) -> CareerMatchingPipeline:
        """Helper to assemble an offline test pipeline with MockLLMProvider."""
        mock_provider = MockLLMProvider(canned_response=mock_response, simulate_error=simulate_error)
        engine = StructuredOutputEngine(dispatcher=mock_provider)
        explainer = MatchExplainer(engine=engine)
        return CareerMatchingPipeline(
            resume_pipeline=ResumePipeline(),
            jd_processor=JobDescriptionProcessor(),
            matching_engine=DeterministicMatchingEngine(),
            explainer=explainer,
        )

    def test_pipeline_with_new_pdf_file_upload(self):
        """1. End-to-end evaluation with a new PDF bytes stream succeeds and attaches explanation."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
            resume_filename="jane_doe_resume.pdf",
        )

        assert isinstance(result, PipelineMatchResult)
        assert result.resume_profile.status == ProfileProcessingStatus.SUCCESS
        assert result.resume_profile.filename == "jane_doe_resume.pdf"
        assert result.jd_profile.job_title == "Senior Backend Engineer"

        # Deterministic match results
        match = result.match_result
        assert match.overall_score > 70.0
        assert match.evaluation_status == EvaluationStatus.FULL_EVALUATION
        assert match.fit_classification in (CandidateFitLevel.STRONG, CandidateFitLevel.EXCEPTIONAL)
        assert len(match.matched_skills) >= 3

        # Explanation attached
        assert match.ai_explanation is not None
        assert "Strong match" in match.ai_explanation.overall_match_summary
        assert match.llm_explanation_metadata["fallback_mode"] is False

    def test_pipeline_with_existing_resume_profile(self):
        """2. End-to-end evaluation using a pre-extracted ResumeProfile (Tab 1 reuse) succeeds without re-parsing."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)
        pre_extracted = ResumePipeline().process_resume(pdf_bytes, filename="cached_jane.pdf")

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_profile=pre_extracted,
        )

        assert result.resume_profile.filename == "cached_jane.pdf"
        assert result.match_result.overall_score > 70.0
        assert result.match_result.evaluation_status == EvaluationStatus.FULL_EVALUATION

    def test_pipeline_rejects_ambiguous_simultaneous_inputs(self):
        """3. Providing both resume_file and resume_profile raises PipelineInputError."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)
        pre_extracted = ResumePipeline().process_resume(pdf_bytes)

        with pytest.raises(PipelineInputError, match="Ambiguous input"):
            pipeline.evaluate(
                jd_text=SAMPLE_JD_FULL,
                resume_file=pdf_bytes,
                resume_profile=pre_extracted,
            )

    def test_pipeline_rejects_missing_inputs(self):
        """4. Providing neither resume_file nor resume_profile raises PipelineInputError."""
        pipeline = self._build_pipeline()

        with pytest.raises(PipelineInputError, match="Missing input"):
            pipeline.evaluate(
                jd_text=SAMPLE_JD_FULL,
                resume_file=None,
                resume_profile=None,
            )

    def test_pipeline_rejects_corrupted_pdf(self):
        """5. Passing invalid or corrupted PDF bytes raises PDFProcessingError."""
        pipeline = self._build_pipeline()
        corrupted_bytes = b"%PDF-1.4 Corrupted content without EOF marker..."

        with pytest.raises(PDFProcessingError):
            pipeline.evaluate(
                jd_text=SAMPLE_JD_FULL,
                resume_file=corrupted_bytes,
                resume_filename="corrupted.pdf",
            )

    def test_pipeline_rejects_empty_or_whitespace_jd(self):
        """6. Passing empty or whitespace-only JD text raises ValueError."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        with pytest.raises(ValueError, match="cannot be empty"):
            pipeline.evaluate(
                jd_text="   \n\t   ",
                resume_file=pdf_bytes,
            )

    def test_pipeline_provisional_jd(self):
        """7. Sparse JD yields PROVISIONAL_EVALUATION and PROVISIONAL fit classification."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_SPARSE,
            resume_file=pdf_bytes,
        )

        assert result.match_result.evaluation_status == EvaluationStatus.PROVISIONAL_EVALUATION
        assert result.match_result.fit_classification == CandidateFitLevel.PROVISIONAL
        assert any("Sparse job description" in w for w in result.match_result.processing_warnings)

    def test_pipeline_single_dimension_unassessed_reserve(self):
        """8. Confirms 40% unassessed reserve is allocated when only 1 dimension is active."""
        pipeline = self._build_pipeline()
        skills_only_resume = (
            "Alice Developer\n\n"
            "TECHNICAL SKILLS\n"
            "Python, FastAPI, Docker, SQL, PostgreSQL, Linux\n"
        )
        pdf_bytes = make_test_pdf(skills_only_resume)
        skills_only_jd = (
            "Requirements:\n"
            "- Python\n"
            "- FastAPI\n"
            "- Docker\n"
            "- PostgreSQL\n"
        )
        result = pipeline.evaluate(
            jd_text=skills_only_jd,
            resume_file=pdf_bytes,
        )
        # Only Technical Skills active -> capped at 60%, unassessed reserve = 40%
        assert result.match_result.unassessed_reserve == 40.0
        assert result.match_result.active_evaluation_coverage == 60.0

    def test_pipeline_insufficient_evidence(self):
        """8. Zero-requirement JD yields INSUFFICIENT_EVIDENCE and 0.0% compatibility score."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_ZERO_REQUIREMENTS,
            resume_file=pdf_bytes,
        )

        assert result.match_result.evaluation_status == EvaluationStatus.INSUFFICIENT_EVIDENCE
        assert result.match_result.overall_score == 0.0

    def test_pipeline_full_evaluation_not_requiring_100_percent(self):
        """9. Confirms FULL_EVALUATION completes with an accurate non-100% score (Refinement 9)."""
        pipeline = self._build_pipeline()
        # Resume lacks AWS, so overall score is realistically bounded below 100%
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
        )

        assert result.match_result.evaluation_status == EvaluationStatus.FULL_EVALUATION
        # Candidate score should reflect realistic technical gaps (not 100%)
        assert 70.0 <= result.match_result.overall_score < 99.0

    def test_pipeline_llm_failure_triggers_deterministic_fallback(self):
        """10. Provider failure does not discard valid deterministic result; triggers fallback (Refinements 3 & 10)."""
        pipeline = self._build_pipeline(simulate_error=LLMRateLimitError("Rate limit exceeded"))
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
        )

        # Deterministic result is fully preserved
        assert result.match_result.overall_score > 70.0
        assert result.match_result.evaluation_status == EvaluationStatus.FULL_EVALUATION
        assert result.match_result.ai_explanation is not None

        # Provenance correctly reflects rule-based fallback
        meta = result.match_result.llm_explanation_metadata
        assert meta["fallback_mode"] is True
        assert meta["provider"] == "rule_based_fallback"
        assert any("Deterministic Rule-Based Fallback" in w for w in result.match_result.ai_explanation.evidence_limitations_and_warnings)

    def test_pipeline_deterministic_immutability(self):
        """11. Verifies scores in final result match DeterministicMatchingEngine output exactly (Refinement 2)."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)
        resume_profile = ResumePipeline().process_resume(pdf_bytes)
        jd_profile = JobDescriptionProcessor().process(SAMPLE_JD_FULL)

        baseline_engine_result = DeterministicMatchingEngine().match(resume_profile, jd_profile)
        pipeline_result = pipeline.evaluate(jd_text=SAMPLE_JD_FULL, resume_profile=resume_profile)

        # Scores must be 100% identical
        assert pipeline_result.match_result.overall_score == baseline_engine_result.overall_score
        assert pipeline_result.match_result.fit_classification == baseline_engine_result.fit_classification
        assert pipeline_result.match_result.active_evaluation_coverage == baseline_engine_result.active_evaluation_coverage
        assert pipeline_result.match_result.unassessed_reserve == baseline_engine_result.unassessed_reserve

    def test_pipeline_json_export_allowlist_privacy(self):
        """12. Confirms allowlist JSON export contains zero raw resume text, contact PII, or secrets (Refinement 4)."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
        )

        export_data = result.to_export_dict()
        export_json = json.dumps(export_data)

        # Must NOT leak raw text or contact details
        assert "jane.doe@example.com" not in export_json
        assert "555" not in export_json
        assert "San Francisco" not in export_json
        assert "Jane Doe" not in export_json
        assert "raw_text" not in export_data
        assert "cleaned_text" not in export_data
        assert "original_text" not in export_data

        # Must include deterministic metrics and evidence
        assert "overall_score" in export_data["deterministic_match"]
        assert "dimensional_breakdown" in export_data["deterministic_match"]
        assert "exact_matches_supported_by_resume" in export_data["skills_and_evidence"]
        assert "qualitative_explanation" in export_data

    def test_pipeline_zero_live_network_calls(self, monkeypatch: pytest.MonkeyPatch):
        """13. Verifies zero live network calls occur during pipeline evaluation (socket isolation)."""
        import socket

        def forbid_socket(*args, **kwargs):
            raise AssertionError("Network socket called during offline pipeline evaluation!")

        monkeypatch.setattr(socket, "socket", forbid_socket)

        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        result = pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
        )
        assert result.match_result.overall_score > 0.0

    def test_pipeline_repeated_analyses_isolation(self):
        """14. Repeated pipeline evaluations with different candidates produce independent results (Refinement 6)."""
        pipeline = self._build_pipeline()

        # Candidate A: Senior Backend
        pdf_a = make_test_pdf(SAMPLE_RESUME_TEXT)
        result_a = pipeline.evaluate(jd_text=SAMPLE_JD_FULL, resume_file=pdf_a, resume_filename="candidate_a.pdf")

        # Candidate B: Junior Frontend
        resume_b_text = (
            "Alex Smith\n"
            "alex@example.com\n\n"
            "SKILLS\nHTML, CSS, JavaScript\n\n"
            "WORK EXPERIENCE\nJunior Web Intern at Startup | 2023 - 2024\n- Built UI mockups.\n"
        )
        pdf_b = make_test_pdf(resume_b_text)
        result_b = pipeline.evaluate(jd_text=SAMPLE_JD_FULL, resume_file=pdf_b, resume_filename="candidate_b.pdf")

        # Confirm results are completely isolated
        assert result_a.resume_profile.filename == "candidate_a.pdf"
        assert result_b.resume_profile.filename == "candidate_b.pdf"
        assert result_a.match_result.overall_score != result_b.match_result.overall_score
        assert result_a.match_result.overall_score > result_b.match_result.overall_score

    def test_pipeline_status_callbacks(self):
        """15. Confirms status_callback receives all stages in sequence (Refinement 8)."""
        pipeline = self._build_pipeline()
        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)

        stages_visited = []

        def callback(stage: str, msg: str):
            stages_visited.append((stage, msg))

        pipeline.evaluate(
            jd_text=SAMPLE_JD_FULL,
            resume_file=pdf_bytes,
            status_callback=callback,
        )

        stage_names = [s[0] for s in stages_visited]
        assert "resume_processing" in stage_names
        assert "jd_processing" in stage_names
        assert "deterministic_matching" in stage_names
        assert "explanation_generation" in stage_names
        assert "complete" in stage_names

    def test_pipeline_fallback_invokes_score_immutability_verification(self, monkeypatch: pytest.MonkeyPatch):
        """16. Confirms fallback path executes verify_result_immutability, preserving all deterministic fields (Correction 1)."""
        import core.matching.pipeline as pipeline_mod
        from core.matching.exceptions import ScoreImmutabilityError

        # Track that verify_result_immutability is called on the pipeline fallback path
        verifier_called = []
        original_verifier = pipeline_mod.verify_result_immutability

        def spy_verifier(orig, candidate):
            verifier_called.append((orig.overall_score, candidate.overall_score))
            return original_verifier(orig, candidate)

        monkeypatch.setattr(pipeline_mod, "verify_result_immutability", spy_verifier)

        # Force explainer.attach_explanation to raise an unhandled exception to trigger pipeline fallback
        pipeline = self._build_pipeline()
        def fail_attach(*args, **kwargs):
            raise RuntimeError("Catastrophic explanation engine failure")
        monkeypatch.setattr(pipeline.explainer, "attach_explanation", fail_attach)

        pdf_bytes = make_test_pdf(SAMPLE_RESUME_TEXT)
        result = pipeline.evaluate(jd_text=SAMPLE_JD_FULL, resume_file=pdf_bytes)

        # Verify fallback occurred and verify_result_immutability was explicitly invoked
        assert result.match_result.llm_explanation_metadata["fallback_mode"] is True
        assert len(verifier_called) >= 1
        assert verifier_called[0][0] == result.match_result.overall_score

        # Confirm every single authoritative field in fallback result matches raw deterministic engine
        submission_meta = {
            "source": "CareerMatchingPipeline",
            "resume_filename": result.resume_profile.filename,
        }
        baseline = DeterministicMatchingEngine().match(
            result.resume_profile,
            result.jd_profile,
            submission_metadata=submission_meta,
        )
        assert result.match_result.overall_score == baseline.overall_score
        assert result.match_result.fit_classification == baseline.fit_classification
        assert result.match_result.evaluation_status == baseline.evaluation_status
        assert result.match_result.active_evaluation_coverage == baseline.active_evaluation_coverage
        assert result.match_result.unassessed_reserve == baseline.unassessed_reserve
        assert result.match_result.match_breakdown == baseline.match_breakdown
        assert result.match_result.matched_skills == baseline.matched_skills
        assert result.match_result.partial_matches == baseline.partial_matches
        assert result.match_result.missing_evidence == baseline.missing_evidence
        assert result.match_result.unknown_evidence == baseline.unknown_evidence
        assert result.match_result.project_evidence == baseline.project_evidence
        assert result.match_result.strengths == baseline.strengths
        assert result.match_result.skill_gaps == baseline.skill_gaps
        assert result.match_result.recommendations == baseline.recommendations
        assert result.match_result.processing_warnings == baseline.processing_warnings
        assert result.match_result.job_metadata == baseline.job_metadata
