import pytest

from src.analysis.metrics import ItemResult, accuracy, delta, gap, weak_agreement_delta, weak_agreement_rate
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


def test_weak_agreement_rate_counts_matches_with_weak_argmax():
    results = [
        ItemResult("q1", "eval", "B", "A"),  # wrong, but matches weak's "B"
        ItemResult("q2", "eval", "C", "C"),  # correct, and matches weak's "C"
        ItemResult("q3", "eval", "A", "D"),  # wrong, does NOT match weak's "B"
        ItemResult("q4", "eval", "D", "A"),  # wrong, weak has no entry for q4
    ]
    weak_argmax = {"q1": "B", "q2": "C", "q3": "B"}
    # 2 of 4 eval predictions match the weak model's own letter (q1, q2).
    assert weak_agreement_rate(results, weak_argmax) == pytest.approx(0.5)


def test_weak_agreement_rate_only_considers_the_given_framing():
    results = [
        ItemResult("q1", "eval", "B", "A"),
        ItemResult("q1", "deploy", "A", "A"),
    ]
    weak_argmax = {"q1": "B"}
    assert weak_agreement_rate(results, weak_argmax, framing="eval") == pytest.approx(1.0)
    assert weak_agreement_rate(results, weak_argmax, framing="deploy") == pytest.approx(0.0)


def test_weak_agreement_rate_empty_framing_raises():
    with pytest.raises(ValueError):
        weak_agreement_rate([ItemResult("q1", "deploy", "A", "A")], {"q1": "A"}, framing="eval")


def test_weak_agreement_delta_isolates_locked_from_control():
    locked = [
        ItemResult("q1", "eval", "B", "A"),
        ItemResult("q2", "eval", "B", "A"),
    ]
    control = [
        ItemResult("q1", "eval", "A", "A"),
        ItemResult("q2", "eval", "C", "A"),
    ]
    weak_argmax = {"q1": "B", "q2": "B"}
    # locked matches weak on both (rate 1.0); control matches on neither (rate 0.0).
    assert weak_agreement_delta(locked, control, weak_argmax) == pytest.approx(1.0)


def test_weak_agreement_delta_zero_when_both_conditions_agree_equally():
    locked = [ItemResult("q1", "eval", "B", "A")]
    control = [ItemResult("q1", "eval", "B", "A")]
    weak_argmax = {"q1": "B"}
    assert weak_agreement_delta(locked, control, weak_argmax) == pytest.approx(0.0)
