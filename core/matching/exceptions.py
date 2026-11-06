"""
Typed exception classes for the deterministic job matching engine.
Preserves explicit error hierarchy without masking system diagnostics.
"""


class MatchingEngineError(Exception):
    """Base exception for all matching engine errors."""
    pass


class InsufficientCriteriaError(MatchingEngineError):
    """Raised when a job description lacks sufficient assessable criteria for evaluation."""
    pass


class InvalidProfileError(MatchingEngineError):
    """Raised when an input resume profile or job description profile fails validation."""
    pass


class ScoreImmutabilityError(MatchingEngineError):
    """Raised when an explanation generator mutates or attempts to alter deterministic matching results."""
    pass


class PipelineInputError(MatchingEngineError):
    """Raised when pipeline input parameters are invalid or ambiguous."""
    pass

