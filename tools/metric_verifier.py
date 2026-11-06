"""
Deterministic tool for validating quantitative claims and metric disclosures.
Delegates directly to MetricGroundingValidator in core.workflow.validators.
"""
from core.workflow.validators import MetricGroundingValidator
from models.workflow import (
    DeterministicValidationViolation,
    MetricDisclosureStatus,
    URLVerificationStatus,
)


class MetricVerifierTool:
    """
    Deterministic tool enforcing disclosure requirements for quantitative claims
    and preventing false third-party verification assertions.
    """

    def __init__(self) -> None:
        self._validator = MetricGroundingValidator()

    def validate_metrics(
        self,
        text: str,
        disclosure_status: MetricDisclosureStatus,
        url_status: URLVerificationStatus = URLVerificationStatus.NO_URL_PROVIDED,
        field_name: str = "metric",
    ) -> list[DeterministicValidationViolation]:
        """
        Check for quantitative claims (percentages, multipliers, currency, latency, scale)
        and verify that SELF_REPORTED_METRIC disclosure is declared.
        Rejects assertions of independent third-party audit/verification.
        """
        return self._validator.validate_metric_claim(
            text=text,
            disclosure_status=disclosure_status,
            url_status=url_status,
            field_name=field_name,
        )

