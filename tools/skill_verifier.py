"""
Deterministic tool for candidate skill-claim verification and advisory grounding.
Delegates directly to CandidateSkillGroundingValidator in core.workflow.validators.
Enforces strict distinction between candidate assertions and forward-looking advisory recommendations.
"""
from typing import Optional
from core.workflow.validators import CandidateSkillGroundingValidator
from models.matching import JobDescriptionProfile
from models.resume_profile import ResumeProfile
from models.workflow import ClaimContext, DeterministicValidationViolation


class SkillVerifierTool:
    """
    Deterministic tool validating technical skill mentions across deliverables.
    Enforces a strict candidate whitelist for historical assertions, while allowing
    forward-looking recommendations to reference missing job description requirements.
    """

    def __init__(self) -> None:
        self._validator = CandidateSkillGroundingValidator()

    def validate_candidate_assertion(
        self,
        skill_name: str,
        profile: ResumeProfile,
        field_name: Optional[str] = None,
    ) -> Optional[DeterministicValidationViolation]:
        """
        Validate a skill claimed as possessed by the candidate.
        Requires evidence in the candidate ResumeProfile or recognized synonyms.
        """
        return self._validator.validate_skill_claim(
            skill_name=skill_name,
            context=ClaimContext.CANDIDATE_ASSERTION,
            profile=profile,
            field_name=field_name,
        )

    def validate_advisory_recommendation(
        self,
        skill_name: str,
        profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
        field_name: Optional[str] = None,
    ) -> Optional[DeterministicValidationViolation]:
        """
        Validate a skill cited in an advisory context (upskilling milestone or learning roadmap).
        Permitted to cite missing JD requirements.
        """
        return self._validator.validate_skill_claim(
            skill_name=skill_name,
            context=ClaimContext.ADVISORY_RECOMMENDATION,
            profile=profile,
            jd_profile=jd_profile,
            field_name=field_name,
        )

    def validate(
        self,
        skill_name: str,
        context: ClaimContext,
        profile: ResumeProfile,
        jd_profile: Optional[JobDescriptionProfile] = None,
        field_name: Optional[str] = None,
    ) -> Optional[DeterministicValidationViolation]:
        """
        General skill validation entry point enforcing the provided ClaimContext.
        """
        return self._validator.validate_skill_claim(
            skill_name=skill_name,
            context=context,
            profile=profile,
            jd_profile=jd_profile,
            field_name=field_name,
        )

    def validate_prose_skills(
        self,
        text: str,
        profile: ResumeProfile,
        field_name: Optional[str] = None,
    ) -> list[DeterministicValidationViolation]:
        """
        Conservative deterministic check for recognizable technical skill terms in bullet prose.
        """
        return self._validator.validate_prose_skill_claims(
            text=text,
            profile=profile,
            field_name=field_name,
        )

