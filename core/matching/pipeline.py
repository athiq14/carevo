"""
End-to-end Career Matching Pipeline orchestrating resume processing,
job description requirement extraction, deterministic candidate scoring,
and evidence-grounded qualitative career explanations.
"""
from datetime import datetime, timezone
import io
from typing import Any, BinaryIO, Callable, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from core.matching.engine import DeterministicMatchingEngine
from core.matching.exceptions import (
    InvalidProfileError,
    MatchingEngineError,
    PipelineInputError,
)
from core.matching.explanation_generator import (
    MatchExplainer,
    scrub_pii,
    synthesize_deterministic_fallback_explanation,
    verify_result_immutability,
)
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


# =====================================================================
# Pipeline Result Container with Allowlist JSON Export
# =====================================================================

class PipelineMatchResult(BaseModel):
    """
    Consolidated output of the CareerMatchingPipeline.
    Encapsulates the authoritative deterministic match result, the parsed
    resume profile, and the normalized job description profile.
    """
    model_config = ConfigDict(frozen=True)

    match_result: JobMatchResult = Field(
        ...,
        description="Authoritative match result with scores, breakdowns, and qualitative explanation.",
    )
    resume_profile: ResumeProfile = Field(
        ...,
        description="Validated unified candidate resume profile.",
    )
    jd_profile: JobDescriptionProfile = Field(
        ...,
        description="Extracted and normalized job description profile.",
    )

    def to_export_dict(self) -> dict[str, Any]:
        """
        Generate a strictly allowlist-filtered JSON-serializable dictionary.
        Guarantees:
        - ZERO raw resume text.
        - ZERO candidate contact information (names, emails, phones, addresses).
        - ZERO API keys, secrets, or internal server paths.
        - Full disclosure of scores, breakdown, evidence, and explanation provenance.
        """
        res = self.match_result
        meta = res.job_metadata or {}
        job_title = scrub_pii(str(meta.get("job_title") or self.jd_profile.job_title or "Target Role"))
        company = scrub_pii(str(meta.get("company") or self.jd_profile.company or "Target Employer"))

        # Dimensional breakdown
        dims = []
        for d in res.match_breakdown:
            dims.append({
                "dimension_name": d.dimension_name,
                "score": round(d.dimension_score, 2),
                "baseline_weight": round(d.baseline_weight, 4),
                "active_weight": round(d.active_weight, 4),
                "evidence_status": d.evidence_status.value,
                "is_active": d.is_active,
                "neutralization_explanation": d.neutralization_explanation,
            })

        # Exact skills supported by resume evidence
        exact_matches = [
            {
                "skill_name": scrub_pii(s.canonical_name),
                "priority": s.requirement_priority.value,
                "evidence_status": s.evidence_status.value,
            }
            for s in res.matched_skills
        ]

        # Related technology matches
        related_matches = [
            {
                "requested_skill": scrub_pii(p.canonical_name),
                "resume_exposure": scrub_pii(p.resume_evidence_references[0]) if p.resume_evidence_references else "Related technology",
                "credit_factor": round(p.match_confidence, 2),
            }
            for p in res.partial_matches
        ]

        # Missing evidence
        missing = [
            {
                "skill_name": scrub_pii(g.skill_name),
                "priority": g.priority.value,
                "suggested_learning_direction": scrub_pii(g.suggested_learning_direction),
            }
            for g in res.missing_evidence
        ]

        # Unknown / qualitative requirements
        unknowns = [
            {
                "requirement": scrub_pii(u.skill_name),
                "priority": u.requirement_priority.value,
                "evidence_status": u.evidence_status.value,
            }
            for u in res.unknown_evidence
        ]

        # Project audit records
        projects = []
        for pe in res.project_evidence:
            projects.append({
                "project_name": scrub_pii(pe.project_name),
                "relevant_technologies": [scrub_pii(t) for t in pe.relevant_technologies],
                "url_verification_status": pe.evidence_verification_level.value,
                "self_reported_metrics": [scrub_pii(m) for m in pe.self_reported_metrics],
            })

        # Qualitative AI explanation if present
        explanation_dict = None
        if res.ai_explanation is not None:
            explanation_dict = {
                "overall_match_summary": res.ai_explanation.overall_match_summary,
                "key_strengths": res.ai_explanation.key_strengths,
                "skill_gaps_assessment": res.ai_explanation.skill_gaps_assessment,
                "experience_assessment": res.ai_explanation.experience_assessment,
                "education_assessment": res.ai_explanation.education_assessment,
                "project_assessment": res.ai_explanation.project_assessment,
                "improvement_recommendations": res.ai_explanation.improvement_recommendations,
                "interview_preparation_suggestions": res.ai_explanation.interview_preparation_suggestions,
                "evidence_limitations_and_warnings": res.ai_explanation.evidence_limitations_and_warnings,
            }

        return {
            "report_metadata": {
                "schema_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "system": "CAREVO AI Career Intelligence Platform - Phase 4",
            },
            "role_context": {
                "job_title": job_title,
                "company": company,
            },
            "deterministic_match": {
                "overall_score": round(res.overall_score, 2),
                "fit_classification": res.fit_classification.value,
                "evaluation_status": res.evaluation_status.value,
                "active_evaluation_coverage": round(res.active_evaluation_coverage, 2),
                "unassessed_reserve": round(res.unassessed_reserve, 2),
                "dimensional_breakdown": dims,
            },
            "skills_and_evidence": {
                "exact_matches_supported_by_resume": exact_matches,
                "related_technology_matches": related_matches,
                "missing_evidence": missing,
                "unassessed_qualitative_requirements": unknowns,
            },
            "projects_audit": {
                "evaluated_projects_count": len(res.project_evidence),
                "projects": projects,
            },
            "qualitative_explanation": explanation_dict,
            "explanation_provenance": res.llm_explanation_metadata or {
                "provider": "unknown",
                "fallback_mode": False,
            },
            "diagnostics_and_warnings": [scrub_pii(w) for w in res.processing_warnings],
            "recommendations": [scrub_pii(r) for r in res.recommendations],
        }


# =====================================================================
# Pipeline Orchestrator Service
# =====================================================================

class CareerMatchingPipeline:
    """
    Orchestrates the complete Career Matching flow:
    1. In-memory Resume extraction and validation.
    2. Job Description parsing and threshold counting.
    3. Authoritative deterministic matching score computation.
    4. Qualitative explanation generation with automatic rule-based fallback.
    """

    def __init__(
        self,
        resume_pipeline: Optional[ResumePipeline] = None,
        jd_processor: Optional[JobDescriptionProcessor] = None,
        matching_engine: Optional[DeterministicMatchingEngine] = None,
        explainer: Optional[MatchExplainer] = None,
    ) -> None:
        self.resume_pipeline = resume_pipeline or ResumePipeline()
        self.jd_processor = jd_processor or JobDescriptionProcessor()
        self.matching_engine = matching_engine or DeterministicMatchingEngine()
        self.explainer = explainer or MatchExplainer()

    def evaluate(
        self,
        jd_text: str,
        resume_file: Optional[Union[bytes, BinaryIO]] = None,
        resume_filename: str = "resume.pdf",
        resume_profile: Optional[ResumeProfile] = None,
        job_title: Optional[str] = None,
        company: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> PipelineMatchResult:
        """
        Execute end-to-end evaluation.

        Args:
            jd_text: Raw job description text.
            resume_file: In-memory bytes or stream of the uploaded resume PDF.
            resume_filename: Name of the uploaded file.
            resume_profile: Pre-extracted ResumeProfile from Tab 1.
            job_title: Optional job title override.
            company: Optional company name override.
            status_callback: Optional callback func(stage_id, message) for UI progress.

        Returns:
            PipelineMatchResult: Validated result containing deterministic evaluation,
                                profiles, and qualitative AI explanation.

        Raises:
            PipelineInputError: If input parameters are ambiguous or missing.
            PDFProcessingError: If uploaded PDF is invalid, corrupted, or encrypted.
            ValueError: If job description text is empty or invalid.
        """
        # -------------------------------------------------------------
        # 1. Input Contract Validation (Refinement 1)
        # -------------------------------------------------------------
        if resume_file is not None and resume_profile is not None:
            raise PipelineInputError(
                "Ambiguous input: Provide either a new resume PDF file or an existing ResumeProfile, not both."
            )
        if resume_file is None and resume_profile is None:
            raise PipelineInputError(
                "Missing input: Provide either an uploaded resume PDF file or an existing ResumeProfile from Session."
            )

        # -------------------------------------------------------------
        # 2. Stage 1: Candidate Resume Processing
        # -------------------------------------------------------------
        active_profile: ResumeProfile
        if resume_file is not None:
            if status_callback:
                status_callback("resume_processing", f"Parsing and validating resume document ({resume_filename})...")
            active_profile = self.resume_pipeline.process_resume(
                file_input=resume_file,
                filename=resume_filename,
                raise_on_error=True,
            )
            if active_profile.status == ProfileProcessingStatus.FAILED:
                err_msg = active_profile.warnings[0] if active_profile.warnings else "Resume document processing failed."
                raise PDFProcessingError(err_msg)
        else:
            if not isinstance(resume_profile, ResumeProfile):
                if (
                    getattr(resume_profile, "__class__", None) is not None
                    and resume_profile.__class__.__name__ == "ResumeProfile"
                    and hasattr(resume_profile, "model_dump")
                ):
                    resume_profile = ResumeProfile.model_validate(resume_profile.model_dump())
                else:
                    raise PipelineInputError(f"Expected ResumeProfile instance, got {type(resume_profile).__name__}.")
            if resume_profile.status == ProfileProcessingStatus.FAILED:
                raise InvalidProfileError("Provided ResumeProfile is in FAILED processing status.")
            active_profile = resume_profile
            if status_callback:
                status_callback("resume_processing", f"Reusing verified candidate profile ({active_profile.filename})...")

        # -------------------------------------------------------------
        # 3. Stage 2: Job Description Parsing & Normalization
        # -------------------------------------------------------------
        if status_callback:
            status_callback("jd_processing", "Extracting and normalizing employer requirements...")

        jd_prof = self.jd_processor.process(
            raw_text=jd_text,
            job_title_override=job_title,
            company_override=company,
        )

        # -------------------------------------------------------------
        # 4. Stage 3: Deterministic Candidate Matching
        # -------------------------------------------------------------
        if status_callback:
            status_callback("deterministic_matching", "Computing deterministic compatibility across 4 evaluation dimensions...")

        submission_meta = {
            "source": "CareerMatchingPipeline",
            "resume_filename": active_profile.filename,
        }
        if job_title:
            submission_meta["job_title"] = job_title
        if company:
            submission_meta["company"] = company

        raw_match_result = self.matching_engine.match(
            resume_profile=active_profile,
            jd_profile=jd_prof,
            submission_metadata=submission_meta,
        )

        # -------------------------------------------------------------
        # 5. Stage 4: Qualitative Explanation Attachment (Refinements 2, 3, 10)
        # -------------------------------------------------------------
        if status_callback:
            status_callback("explanation_generation", "Synthesizing evidence-grounded career explanation...")

        final_match_result: JobMatchResult
        try:
            final_match_result = self.explainer.attach_explanation(
                match_result=raw_match_result,
                fallback_on_error=True,
            )
        except Exception as exc:
            # Absolute resilience: Explanation failure must never discard a valid deterministic matching result
            fallback_explanation = synthesize_deterministic_fallback_explanation(
                raw_match_result,
                failure_reason=f"Pipeline explanation error: {exc}",
            )
            fallback_meta = {
                "provider": "rule_based_fallback",
                "model": "deterministic_template",
                "error": str(exc),
                "fallback_mode": True,
            }
            final_match_result = raw_match_result.model_copy(
                update={
                    "ai_explanation": fallback_explanation,
                    "llm_explanation_metadata": fallback_meta,
                },
                deep=True,
            )
            # Strict deterministic immutability verification on fallback path (operational under python -O)
            verify_result_immutability(raw_match_result, final_match_result)

        if status_callback:
            status_callback("complete", "Career matching analysis completed successfully.")

        return PipelineMatchResult(
            match_result=final_match_result,
            resume_profile=active_profile,
            jd_profile=jd_prof,
        )

