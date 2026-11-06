"""
Deterministic Job Matching and Compatibility Engine package.
Provides modular evaluators for technical skills, work experience, academic education,
and practical engineering projects, unified under DeterministicMatchingEngine.
"""
from core.matching.education_evaluator import (
    ADJACENT_STEM_FIELDS,
    CORE_STEM_FIELDS,
    EducationEvaluator,
)
from core.matching.engine import DeterministicMatchingEngine
from core.matching.exceptions import (
    InsufficientCriteriaError,
    InvalidProfileError,
    MatchingEngineError,
    PipelineInputError,
    ScoreImmutabilityError,
)
from core.matching.explanation_generator import (
    MatchExplainer,
    build_sanitized_explanation_payload,
    escape_xml_closing_tags,
    scrub_pii,
    synthesize_deterministic_fallback_explanation,
    verify_result_immutability,
)
from core.matching.pipeline import (
    CareerMatchingPipeline,
    PipelineMatchResult,
)
from core.matching.experience_evaluator import (
    TIER_SCORES,
    ExperienceEvaluator,
)
from core.matching.project_evaluator import (
    QUANTIFIED_METRIC_PATTERN,
    ProjectEvaluator,
)
from core.matching.skill_matcher import (
    RELATED_TECH_MAP,
    SkillMatcher,
)
from core.matching.weight_manager import (
    BASELINE_WEIGHTS,
    MAX_DIMENSION_CAP,
    WeightManager,
)

__all__ = [
    "DeterministicMatchingEngine",
    "SkillMatcher",
    "ExperienceEvaluator",
    "EducationEvaluator",
    "ProjectEvaluator",
    "WeightManager",
    "MatchExplainer",
    "CareerMatchingPipeline",
    "PipelineMatchResult",
    "build_sanitized_explanation_payload",
    "synthesize_deterministic_fallback_explanation",
    "scrub_pii",
    "escape_xml_closing_tags",
    "verify_result_immutability",
    "MatchingEngineError",
    "InsufficientCriteriaError",
    "InvalidProfileError",
    "ScoreImmutabilityError",
    "PipelineInputError",
    "BASELINE_WEIGHTS",
    "MAX_DIMENSION_CAP",
    "RELATED_TECH_MAP",
    "TIER_SCORES",
    "CORE_STEM_FIELDS",
    "ADJACENT_STEM_FIELDS",
    "QUANTIFIED_METRIC_PATTERN",
]

