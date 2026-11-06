"""
Deterministic tool for detecting PII leakage in agent deliverables.
Delegates directly to PIILeakageValidator in core.workflow.validators.
"""
from core.workflow.validators import PIILeakageValidator
from models.workflow import DeterministicValidationViolation


class PIIScannerTool:
    """
    Deterministic tool scanning text for candidate personal identifiers
    (emails, phone numbers, street addresses, SSNs) and emitting sanitized
    violations without echoing raw PII.
    """

    def __init__(self) -> None:
        self._validator = PIILeakageValidator()

    def scan(
        self,
        text: str,
        field_name: str = "content",
    ) -> list[DeterministicValidationViolation]:
        """
        Scan text for PII leakage. Returns list of detected violations with redacted invalid values.
        """
        return self._validator.validate_text(text=text, field_name=field_name)

