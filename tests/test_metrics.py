import pytest

from src.analysis.metrics import accuracy, delta, gap
from tests.helpers import make_framing_results, make_model_results


def test_accuracy_all_correct():
    results = make_framing_results("eval", 10, 10)
    assert accuracy(results) == 1.0


def test_accuracy_all_wrong():
    results = make_framing_results("eval", 0, 10)
    assert accuracy(results) == 0.0


def test_accuracy_known_fraction():
    results = make_framing_results("eval", 3, 4)
    assert accuracy(results) == pytest.approx(0.75)


def test_accuracy_empty_raises():
    with pytest.raises(ValueError):
        accuracy([])


def test_gap_known_value():
    # eval: 70/100 correct, deploy: 85/100 correct -> gap should be exactly 15pp.
    results = make_model_results(70, 100, 85, 100)
    assert gap(results) == pytest.approx(0.15)


def test_gap_zero_when_framings_equal():
    results = make_model_results(60, 100, 60, 100)
    assert gap(results) == pytest.approx(0.0)


def test_gap_can_be_negative():
    results = make_model_results(80, 100, 50, 100)
    assert gap(results) == pytest.approx(-0.30)


def test_delta_known_value():
    locked = make_model_results(70, 100, 85, 100)  # gap = 15pp
    control = make_model_results(60, 100, 62, 100)  # gap = 2pp
    assert delta(locked, control) == pytest.approx(0.13)


def test_delta_zero_when_gaps_equal():
    locked = make_model_results(70, 100, 85, 100)  # gap = 15pp
    control = make_model_results(55, 100, 70, 100)  # gap = 15pp
    assert delta(locked, control) == pytest.approx(0.0)
