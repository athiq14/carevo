"""
Deterministic anti-inflation weight manager for candidate-job matching.
Enforces baseline dimensional weights, proportional redistribution,
a strict 60% single-dimension ceiling, and deterministic unassessed reserve allocation.
"""
from typing import Sequence, Union


# Standard Phase 4 evaluation baseline weights
BASELINE_WEIGHTS: dict[str, float] = {
    "skills": 0.45,
    "experience": 0.25,
    "education": 0.15,
    "projects": 0.15,
}

# Maximum allowed weight ceiling for any single active evaluation dimension
MAX_DIMENSION_CAP: float = 0.60


class WeightManager:
    """
    Manages deterministic redistribution of evaluation dimension weights.
    Prevents artificial score inflation when employer job descriptions omit criteria.
    """

    def __init__(
        self,
        baseline_weights: dict[str, float] = BASELINE_WEIGHTS,
        max_dimension_cap: float = MAX_DIMENSION_CAP,
    ) -> None:
        self.baseline_weights = dict(baseline_weights)
        self.max_dimension_cap = max_dimension_cap

    def calculate_weights(
        self,
        active_dimensions: Union[Sequence[str], set[str]],
    ) -> tuple[dict[str, float], float, float]:
        """
        Calculate final active weights using the approved deterministic capped-weight redistribution algorithm.

        Args:
            active_dimensions: Dimensions with assessable JD criteria ('skills', 'experience', 'education', 'projects').

        Returns:
            tuple[dict[str, float], float, float]:
                - active_weights: Dict mapping dimension name to final active weight [0.0, 0.60].
                - active_coverage: Percentage of criteria assessable [0.0, 100.0].
                - unassessed_reserve: Percentage of unassessed criteria held in reserve [0.0, 100.0].
        """
        active_set = {d.strip().lower() for d in active_dimensions if d.strip().lower() in self.baseline_weights}
        all_dims = list(self.baseline_weights.keys())

        # Case 0: No active dimensions (zero assessable criteria)
        if not active_set:
            zero_weights = {d: 0.0 for d in all_dims}
            return zero_weights, 0.0, 100.0

        # Step 1: Compute unconstrained proportional weights
        total_active_baseline = sum(self.baseline_weights[d] for d in active_set)
        if total_active_baseline <= 0.0:
            zero_weights = {d: 0.0 for d in all_dims}
            return zero_weights, 0.0, 100.0

        initial_weights: dict[str, float] = {}
        for d in active_set:
            initial_weights[d] = self.baseline_weights[d] / total_active_baseline

        # Step 2: Check for ceiling violation (at most one dimension can exceed 0.60)
        violating_dim: str | None = None
        for d, w in initial_weights.items():
            if w > self.max_dimension_cap + 1e-9:
                violating_dim = d
                break

        final_active_weights: dict[str, float] = {d: 0.0 for d in all_dims}
        reserve: float = 0.0

        if violating_dim is None:
            # All initial weights <= 0.60
            for d in active_set:
                final_active_weights[d] = initial_weights[d]
            reserve = 0.0
        else:
            # Cap the violating dimension at 0.60
            final_active_weights[violating_dim] = self.max_dimension_cap
            r_rem = 1.00 - self.max_dimension_cap  # 0.40
            other_active = active_set - {violating_dim}

            if not other_active:
                # N = 1 active dimension: remainder is placed into unassessed reserve
                reserve = r_rem  # 0.40
            else:
                # N >= 2: distribute 0.40 proportionally across remaining active dimensions
                sum_other_baseline = sum(self.baseline_weights[o] for o in other_active)
                for o in other_active:
                    final_active_weights[o] = r_rem * (self.baseline_weights[o] / sum_other_baseline)
                reserve = 0.0

        # Precision normalization: ensure sum of active weights + reserve is exactly 1.00
        total_allocated = sum(final_active_weights.values()) + reserve
        if abs(total_allocated - 1.00) > 1e-9:
            # Minor floating point adjustment
            diff = 1.00 - total_allocated
            if active_set:
                first_active = next(iter(active_set))
                final_active_weights[first_active] += diff

        active_coverage = sum(final_active_weights.values()) * 100.0
        unassessed_reserve_pct = reserve * 100.0

        return final_active_weights, active_coverage, unassessed_reserve_pct

