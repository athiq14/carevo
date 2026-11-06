"""
Deterministic tool for validating employment tenure, organization, and job title fidelity.
Delegates directly to EntityTenureValidator in core.workflow.validators.
"""
from typing import Any, Optional
from core.workflow.validators import EntityTenureValidator
from models.resume_entities import WorkExperience
from models.workflow import DeterministicValidationViolation


class TenureVerifierTool:
    """
    Deterministic tool ensuring employment metadata matches source candidate records verbatim
    and preventing academic or project experience from being represented as full-time employment.
    """

    def __init__(self) -> None:
        self._validator = EntityTenureValidator()

    def validate_experience_metadata(
        self,
        source_exp: WorkExperience,
        claimed_title: Optional[str] = None,
        claimed_company: Optional[str] = None,
        claimed_dates: Optional[str] = None,
        field_prefix: str = "experience",
    ) -> list[DeterministicValidationViolation]:
        """
        Verify that employment job titles, employer company names, and dates match source records verbatim.
        """
        return self._validator.validate_work_experience_metadata(
            source_exp=source_exp,
            claimed_title=claimed_title,
            claimed_company=claimed_company,
            claimed_dates=claimed_dates,
            field_prefix=field_prefix,
        )

    def validate_no_project_as_employment(
        self,
        source_record: Any,
        claimed_as_employment: bool,
        field_name: str = "tenure",
    ) -> Optional[DeterministicValidationViolation]:
        """
        Prevent project or academic records from being represented as full-time employment tenure.
        """
        return self._validator.validate_no_project_as_employment(
            source_record=source_record,
            claimed_as_employment=claimed_as_employment,
            field_name=field_name,
        )

