"""
Deterministic tool for Authoritative Phase 4 score integrity and baseline protection.
Delegates directly to score verification functions in core.workflow.validators.
"""
from typing import Any, Optional, Union
from core.workflow.validators import (
    compute_match_result_digest,
    create_baseline_snapshot,
    intercept_node_update,
    serialize_authoritative_match_result,
    verify_workflow_score_immutability,
)
from models.matching import JobMatchResult
from models.workflow import AuthoritativeMatchSnapshot


class ScoreIntegrityTool:
    """
    Deterministic tool protecting Phase 4 authoritative scores, dimensional breakdowns,
    and evidence structures against mutation or tampering during agentic workflows.
    """

    @staticmethod
    def create_snapshot(match_result: JobMatchResult) -> AuthoritativeMatchSnapshot:
        """Create an independent, deeply immutable AuthoritativeMatchSnapshot."""
        return create_baseline_snapshot(match_result)

    @staticmethod
    def compute_digest(result: Union[JobMatchResult, AuthoritativeMatchSnapshot, str]) -> str:
        """Compute the deterministic SHA-256 digest of the 16 authoritative fields."""
        return compute_match_result_digest(result)

    @staticmethod
    def serialize_canonical(result: Any) -> str:
        """Deterministically serialize the 16 authoritative fields to compact JSON."""
        return serialize_authoritative_match_result(result)

    @staticmethod
    def verify_immutability(
        baseline: Union[AuthoritativeMatchSnapshot, JobMatchResult],
        candidate: JobMatchResult,
        expected_sha256: Optional[str] = None,
    ) -> None:
        """
        Verify that candidate preserves all 16 authoritative fields against baseline
        and matches the expected cryptographic SHA-256 digest.
        """
        verify_workflow_score_immutability(
            baseline=baseline,
            candidate=candidate,
            expected_sha256=expected_sha256,
        )

    @staticmethod
    def intercept_update(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
        """Intercept and block node returns attempting to overwrite protected baseline fields."""
        return intercept_node_update(node_name=node_name, update=update)

