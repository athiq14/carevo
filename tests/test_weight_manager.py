"""
Unit tests for core/matching/weight_manager.py.
Verifies the deterministic capped-weight redistribution algorithm across
one, two, three, four, and zero active dimensions, ceiling enforcement, and reserve math.
"""
import pytest

from core.matching.weight_manager import (
    BASELINE_WEIGHTS,
    MAX_DIMENSION_CAP,
    WeightManager,
)


class TestWeightManager:
    """Tests deterministic anti-inflation weight manager."""

    @pytest.fixture
    def manager(self) -> WeightManager:
        return WeightManager()

    def test_four_active_dimensions_preserves_baseline_weights(self, manager: WeightManager):
        """When all 4 dimensions are active, baseline weights are preserved with zero reserve."""
        active = ["skills", "experience", "education", "projects"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["skills"], rel=1e-4) == 0.45
        assert pytest.approx(weights["experience"], rel=1e-4) == 0.25
        assert pytest.approx(weights["education"], rel=1e-4) == 0.15
        assert pytest.approx(weights["projects"], rel=1e-4) == 0.15

        assert pytest.approx(coverage, rel=1e-4) == 100.0
        assert pytest.approx(reserve, rel=1e-4) == 0.0
        assert pytest.approx(sum(weights.values()) + (reserve / 100.0), rel=1e-6) == 1.0

    def test_three_active_dimensions_proportional_redistribution(self, manager: WeightManager):
        """When 1 dimension is neutralized, remaining 3 absorb weight proportionally below the 60% cap."""
        # Projects neutralized: active = skills (0.45), experience (0.25), education (0.15)
        # Total active baseline = 0.85
        # Skills: 0.45 / 0.85 = 0.5294 <= 0.60
        # Exp: 0.25 / 0.85 = 0.2941 <= 0.60
        # Edu: 0.15 / 0.85 = 0.1765 <= 0.60
        active = ["skills", "experience", "education"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["skills"], rel=1e-4) == 0.45 / 0.85
        assert pytest.approx(weights["experience"], rel=1e-4) == 0.25 / 0.85
        assert pytest.approx(weights["education"], rel=1e-4) == 0.15 / 0.85
        assert weights["projects"] == 0.0

        assert pytest.approx(coverage, rel=1e-4) == 100.0
        assert pytest.approx(reserve, rel=1e-4) == 0.0
        assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0

    def test_two_active_dimensions_caps_at_sixty_percent(self, manager: WeightManager):
        """When Skills and Experience only are active, Skills is capped at 0.60 and Exp absorbs 0.40."""
        # Scenario 3: Skills (0.45) + Experience (0.25). Total = 0.70.
        # Initial Skills: 0.45 / 0.70 = 0.6428 > 0.60 -> Capped at 0.60.
        # Remainder 0.40 goes to Experience.
        active = ["skills", "experience"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["skills"], rel=1e-4) == 0.60
        assert pytest.approx(weights["experience"], rel=1e-4) == 0.40
        assert weights["education"] == 0.0
        assert weights["projects"] == 0.0

        assert pytest.approx(coverage, rel=1e-4) == 100.0
        assert pytest.approx(reserve, rel=1e-4) == 0.0
        assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0

    def test_one_active_dimension_caps_at_sixty_and_allocates_reserve(self, manager: WeightManager):
        """When only 1 dimension is active, it is capped at 0.60 and 0.40 is placed in unassessed reserve."""
        active = ["skills"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["skills"], rel=1e-4) == 0.60
        assert weights["experience"] == 0.0
        assert weights["education"] == 0.0
        assert weights["projects"] == 0.0

        assert pytest.approx(coverage, rel=1e-4) == 60.0
        assert pytest.approx(reserve, rel=1e-4) == 40.0
        assert pytest.approx(sum(weights.values()) + (reserve / 100.0), rel=1e-6) == 1.0

    def test_single_experience_dimension_reserve(self, manager: WeightManager):
        """When only experience is active, it is capped at 0.60 with 40% reserve."""
        active = ["experience"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["experience"], rel=1e-4) == 0.60
        assert pytest.approx(reserve, rel=1e-4) == 40.0
        assert pytest.approx(coverage, rel=1e-4) == 60.0

    def test_zero_active_dimensions_returns_hundred_percent_reserve(self, manager: WeightManager):
        """When no dimensions are active, coverage is 0% and reserve is 100%."""
        active: list[str] = []
        weights, coverage, reserve = manager.calculate_weights(active)

        assert all(w == 0.0 for w in weights.values())
        assert coverage == 0.0
        assert reserve == 100.0

    def test_handles_whitespace_and_casing(self, manager: WeightManager):
        """Verifies case-insensitivity and whitespace trimming on dimension inputs."""
        active = ["  SKILLS  ", "EXPERIENCE"]
        weights, coverage, reserve = manager.calculate_weights(active)

        assert pytest.approx(weights["skills"], rel=1e-4) == 0.60
        assert pytest.approx(weights["experience"], rel=1e-4) == 0.40
        assert coverage == 100.0
        assert reserve == 0.0

