"""
Deterministic tool for resolving evidence references against candidate ResumeProfile.
Delegates directly to EvidenceReferenceResolver in core.workflow.validators.
"""
from typing import Any, Optional
from core.workflow.validators import EvidenceReferenceResolver
from models.resume_profile import ResumeProfile
from models.workflow import DeterministicValidationViolation, ProvenanceTier


class EvidenceResolverTool:
    """
    Deterministic tool validating that entity references and field paths
    resolve to actual candidate experience, project, education, or skill records.
    """

    def __init__(self) -> None:
        self._resolver = EvidenceReferenceResolver()

    def resolve(
        self,
        entity_id: str,
        field_path: str,
        profile: ResumeProfile,
    ) -> tuple[bool, Optional[str], Optional[Any]]:
        """
        Resolve an entity ID (e.g. 'exp_0', 'proj_0') and field path against ResumeProfile.
        Returns (is_resolved, error_message, resolved_value).
        """
        return self._resolver.resolve_reference(
            entity_id=entity_id,
            field_path=field_path,
            profile=profile,
        )

    def validate_provenance_tier(
        self,
        tier: ProvenanceTier,
        source_path: str,
    ) -> Optional[DeterministicValidationViolation]:
        """
        Ensures user clarifications cannot be falsely tagged as resume-extracted evidence.
        """
        return self._resolver.validate_provenance_tier(tier=tier, source_path=source_path)

