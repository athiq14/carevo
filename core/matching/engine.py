"""
Deterministic Job Matching Engine.
Orchestrates multi-dimensional candidate evaluation across technical skills,
work experience tenure and relevance, academic qualifications, and practical engineering projects.
Calculates objective compatibility scores, enforces anti-inflation capped redistribution,
and traces every score to verified source evidence.
"""
from typing import Any, Optional

from core.matching.education_evaluator import EducationEvaluator
from core.matching.exceptions import (
    InsufficientCriteriaError,
    InvalidProfileError,
)
from core.matching.experience_evaluator import ExperienceEvaluator
from core.matching.project_evaluator import ProjectEvaluator
from core.matching.skill_matcher import SkillMatcher
from core.matching.weight_manager import BASELINE_WEIGHTS, WeightManager
from models.matching import (
    CandidateFitLevel,
    EvaluationStatus,
    EvidenceStatus,
    JobDescriptionProfile,
    JobMatchResult,
    MatchBreakdown,
    SkillGap,
    SkillMatch,
)
from models.resume_profile import ResumeProfile


class DeterministicMatchingEngine:
    """
    Main evaluation engine for candidate-job matching.
    Coordinates dimensional sub-evaluators and executes deterministic score calculation.
    """

    def __init__(
        self,
        skill_matcher: Optional[SkillMatcher] = None,
        experience_evaluator: Optional[ExperienceEvaluator] = None,
        education_evaluator: Optional[EducationEvaluator] = None,
        project_evaluator: Optional[ProjectEvaluator] = None,
        weight_manager: Optional[WeightManager] = None,
    ) -> None:
        self.skill_matcher = skill_matcher or SkillMatcher()
        self.experience_evaluator = experience_evaluator or ExperienceEvaluator()
        self.education_evaluator = education_evaluator or EducationEvaluator()
        self.project_evaluator = project_evaluator or ProjectEvaluator()
        self.weight_manager = weight_manager or WeightManager()

    def match(
        self,
        resume_profile: ResumeProfile,
        jd_profile: JobDescriptionProfile,
        submission_metadata: Optional[dict[str, Any]] = None,
    ) -> JobMatchResult:
        """
        Evaluate candidate ResumeProfile against JobDescriptionProfile.

        Args:
            resume_profile: Processed, validated resume profile.
            jd_profile: Processed, normalized job description profile.
            submission_metadata: Optional supplementary submission parameters.

        Returns:
            JobMatchResult: Consolidated immutable match evaluation result.
        """
        if not isinstance(resume_profile, ResumeProfile):
            if (
                getattr(resume_profile, "__class__", None) is not None
                and resume_profile.__class__.__name__ == "ResumeProfile"
                and hasattr(resume_profile, "model_dump")
            ):
                resume_profile = ResumeProfile.model_validate(resume_profile.model_dump())
            else:
                raise InvalidProfileError(f"Expected ResumeProfile, got {type(resume_profile).__name__}.")
        if not isinstance(jd_profile, JobDescriptionProfile):
            raise InvalidProfileError(f"Expected JobDescriptionProfile, got {type(jd_profile).__name__}.")

        metadata = dict(submission_metadata or {})
        if jd_profile.job_title:
            metadata.setdefault("job_title", jd_profile.job_title)
        if jd_profile.company:
            metadata.setdefault("company", jd_profile.company)

        warnings: list[str] = list(jd_profile.processing_warnings)

        # Check for Zero Assessable Criteria in JD
        total_jd_reqs = jd_profile.requirement_counts.get("total_requirements", len(jd_profile.normalized_requirements))
        if total_jd_reqs == 0:
            warnings.append(
                "Job description contains zero assessable requirements. Matching cannot evaluate candidate compatibility."
            )
            # Create neutralized breakdowns
            breakdowns = [
                MatchBreakdown(
                    dimension_name="Technical Skills",
                    baseline_weight=BASELINE_WEIGHTS["skills"],
                    active_weight=0.0,
                    dimension_score=0.0,
                    evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                    is_active=False,
                    neutralization_explanation="Zero technical skills specified in job description.",
                ),
                MatchBreakdown(
                    dimension_name="Work Experience",
                    baseline_weight=BASELINE_WEIGHTS["experience"],
                    active_weight=0.0,
                    dimension_score=0.0,
                    evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                    is_active=False,
                    neutralization_explanation="Zero experience criteria specified in job description.",
                ),
                MatchBreakdown(
                    dimension_name="Academic Education",
                    baseline_weight=BASELINE_WEIGHTS["education"],
                    active_weight=0.0,
                    dimension_score=0.0,
                    evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                    is_active=False,
                    neutralization_explanation="Zero education criteria specified in job description.",
                ),
                MatchBreakdown(
                    dimension_name="Engineering Projects",
                    baseline_weight=BASELINE_WEIGHTS["projects"],
                    active_weight=0.0,
                    dimension_score=0.0,
                    evidence_status=EvidenceStatus.UNKNOWN_EVIDENCE,
                    is_active=False,
                    neutralization_explanation="Zero project criteria specified in job description.",
                ),
            ]
            return JobMatchResult(
                job_metadata=metadata,
                overall_score=0.0,
                evaluation_status=EvaluationStatus.INSUFFICIENT_EVIDENCE,
                fit_classification=CandidateFitLevel.LOW,
                active_evaluation_coverage=0.0,
                unassessed_reserve=100.0,
                match_breakdown=breakdowns,
                matched_skills=[],
                partial_matches=[],
                missing_evidence=[],
                unknown_evidence=[],
                project_evidence=[],
                strengths=[],
                skill_gaps=[],
                recommendations=["Provide a job description with concrete technical qualifications and requirements."],
                processing_warnings=warnings,
            )

        # Step 1: Run Sub-Evaluators
        skill_res = self.skill_matcher.evaluate(resume_profile, jd_profile)
        exp_res = self.experience_evaluator.evaluate(resume_profile, jd_profile)
        edu_res = self.education_evaluator.evaluate(resume_profile, jd_profile)

        cand_ft_years = float(exp_res.get("candidate_tenure_years", 0.0))
        proj_res = self.project_evaluator.evaluate(
            resume_profile, jd_profile, candidate_ft_years=cand_ft_years
        )

        # Step 2: Determine Active Dimensions
        active_dimensions: list[str] = []
        if skill_res["is_active"]:
            active_dimensions.append("skills")
        if exp_res["is_active"]:
            active_dimensions.append("experience")
        if edu_res["is_active"]:
            active_dimensions.append("education")
        if proj_res["is_active"]:
            active_dimensions.append("projects")

        # Step 3: Compute Anti-Inflation Weights via WeightManager
        active_weights, active_coverage, unassessed_reserve = self.weight_manager.calculate_weights(
            active_dimensions
        )

        # Step 4: Calculate Overall Weighted Compatibility Score
        score_skills = float(skill_res["dimension_score"])
        score_exp = float(exp_res["dimension_score"])
        score_edu = float(edu_res["dimension_score"])
        score_proj = float(proj_res["dimension_score"])

        w_skills = active_weights.get("skills", 0.0)
        w_exp = active_weights.get("experience", 0.0)
        w_edu = active_weights.get("education", 0.0)
        w_proj = active_weights.get("projects", 0.0)

        overall_score = (
            w_skills * score_skills
            + w_exp * score_exp
            + w_edu * score_edu
            + w_proj * score_proj
        )
        overall_score = max(0.0, min(100.0, overall_score))
        overall_score = round(overall_score, 3)

        # Step 5: Determine Evaluation Lifecycle Status & Fit Level
        is_sparse = (
            jd_profile.is_provisional
            or jd_profile.technical_skill_count < 3
            or total_jd_reqs < 4
        )

        if is_sparse:
            eval_status = EvaluationStatus.PROVISIONAL_EVALUATION
            fit_level = CandidateFitLevel.PROVISIONAL
            warnings.append(
                f"Sparse job description criteria ({jd_profile.technical_skill_count} skills, {total_jd_reqs} total requirements). "
                "Compatibility score is provisional and reflects limited employer specifications."
            )
        else:
            eval_status = EvaluationStatus.FULL_EVALUATION
            if overall_score >= 90.0:
                fit_level = CandidateFitLevel.EXCEPTIONAL
            elif overall_score >= 75.0:
                fit_level = CandidateFitLevel.STRONG
            elif overall_score >= 60.0:
                fit_level = CandidateFitLevel.MODERATE
            elif overall_score >= 40.0:
                fit_level = CandidateFitLevel.DEVELOPING
            else:
                fit_level = CandidateFitLevel.LOW

        # Step 6: Construct Dimensional Breakdown Records
        breakdowns = [
            MatchBreakdown(
                dimension_name="Technical Skills",
                baseline_weight=BASELINE_WEIGHTS["skills"],
                active_weight=round(w_skills, 4),
                dimension_score=round(score_skills, 2),
                evidence_status=(
                    skill_res.get("evidence_status", EvidenceStatus.UNKNOWN_EVIDENCE)
                    if not skill_res["is_active"]
                    else (
                        EvidenceStatus.VERIFIED_EVIDENCE
                        if score_skills >= 75.0
                        else (EvidenceStatus.PARTIAL_EVIDENCE if score_skills > 0.0 else EvidenceStatus.MISSING_EVIDENCE)
                    )
                ),
                is_active=skill_res["is_active"],
                neutralization_explanation=skill_res["neutralization_explanation"],
            ),
            MatchBreakdown(
                dimension_name="Work Experience",
                baseline_weight=BASELINE_WEIGHTS["experience"],
                active_weight=round(w_exp, 4),
                dimension_score=round(score_exp, 2),
                evidence_status=exp_res["evidence_status"],
                is_active=exp_res["is_active"],
                neutralization_explanation=exp_res["neutralization_explanation"],
            ),
            MatchBreakdown(
                dimension_name="Academic Education",
                baseline_weight=BASELINE_WEIGHTS["education"],
                active_weight=round(w_edu, 4),
                dimension_score=round(score_edu, 2),
                evidence_status=edu_res["evidence_status"],
                is_active=edu_res["is_active"],
                neutralization_explanation=edu_res["neutralization_explanation"],
            ),
            MatchBreakdown(
                dimension_name="Engineering Projects",
                baseline_weight=BASELINE_WEIGHTS["projects"],
                active_weight=round(w_proj, 4),
                dimension_score=round(score_proj, 2),
                evidence_status=proj_res["evidence_status"],
                is_active=proj_res["is_active"],
                neutralization_explanation=proj_res["neutralization_explanation"],
            ),
        ]

        # Step 7: Synthesize Strengths, Gaps, and Recommendations
        strengths: list[str] = []
        for sm in skill_res["matched_skills"]:
            strengths.append(f"Demonstrated verified proficiency in {sm.canonical_name}.")
        if exp_res.get("assigned_tier") in ("DIRECT_ALIGNMENT", "ADJACENT_ALIGNMENT"):
            strengths.append(f"Direct role domain alignment in employment history ({exp_res.get('assigned_tier')}).")
        if proj_res.get("c_tech", 0.0) >= 30.0:
            strengths.append("High practical technology coverage across engineering projects.")

        skill_gaps: list[SkillGap] = list(skill_res["missing_evidence"])

        recommendations: list[str] = []
        for gap in skill_gaps[:3]:
            if gap.suggested_learning_direction:
                recommendations.append(f"Bridge {gap.skill_name} gap: {gap.suggested_learning_direction}")
        if unassessed_reserve > 0.0:
            recommendations.append(
                f"Employer job description left {unassessed_reserve:.1f}% of criteria unstated; clarify missing role expectations."
            )

        return JobMatchResult(
            job_metadata=metadata,
            overall_score=overall_score,
            evaluation_status=eval_status,
            fit_classification=fit_level,
            active_evaluation_coverage=round(active_coverage, 2),
            unassessed_reserve=round(unassessed_reserve, 2),
            match_breakdown=breakdowns,
            matched_skills=skill_res["matched_skills"],
            partial_matches=skill_res["partial_matches"],
            missing_evidence=skill_res["missing_evidence"],
            unknown_evidence=skill_res["unknown_evidence"],
            project_evidence=proj_res["project_evidence"],
            strengths=strengths,
            skill_gaps=skill_gaps,
            recommendations=recommendations,
            processing_warnings=warnings,
        )

