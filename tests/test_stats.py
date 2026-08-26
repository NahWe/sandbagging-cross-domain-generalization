import pytest

from src.analysis.stats import bootstrap_delta_ci
from tests.helpers import make_model_results


def test_bootstrap_ci_point_estimate_matches_delta():
    locked = make_model_results(70, 100, 85, 100)  # gap = 15pp
    control = make_model_results(60, 100, 62, 100)  # gap = 2pp
    point, lo, hi = bootstrap_delta_ci(locked, control, n_boot=2000, seed=42)
    assert point == pytest.approx(0.13)
    assert lo <= point <= hi


def test_bootstrap_ci_excludes_zero_for_large_n_and_clear_effect():
    # Large n, clearly different gaps -> CI should confidently exclude zero.
    locked = make_model_results(700, 1000, 850, 1000)  # gap = 15pp
    control = make_model_results(600, 1000, 605, 1000)  # gap = 0.5pp
    _, lo, hi = bootstrap_delta_ci(locked, control, n_boot=2000, seed=42)
    assert lo > 0


def test_bootstrap_ci_includes_zero_when_gaps_equal():
    locked = make_model_results(70, 100, 85, 100)  # gap = 15pp
    control = make_model_results(55, 100, 70, 100)  # gap = 15pp -> delta = 0
    _, lo, hi = bootstrap_delta_ci(locked, control, n_boot=2000, seed=42)
    assert lo <= 0 <= hi
