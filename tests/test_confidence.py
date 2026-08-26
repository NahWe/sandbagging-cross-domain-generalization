import math

import pytest

from src.analysis.confidence import (
    CHOICE_LETTERS,
    ItemDistribution,
    bootstrap_letter_bias_correlation,
    correlate_letter_bias,
    entropy,
    letter_preference_delta,
    mean_entropy,
    rank_distribution_test,
    true_answer_rank,
    wrong_answer_letter_distribution,
)


def _wrong_item(item_id: str, predicted: str, correct: str = "A", framing: str = "eval") -> ItemDistribution:
    """Builds an ItemDistribution whose argmax is `predicted` (correct must
    differ from predicted for downstream "wrong item" filtering to treat it
    as incorrect, matching how a real model's forced-choice logits would
    produce this shape)."""
    assert predicted != correct
    probs = {letter: 0.1 for letter in CHOICE_LETTERS}
    probs[predicted] = 0.7
    return ItemDistribution(
        item_id=item_id, framing=framing, correct=correct, probs=tuple(probs[l] for l in CHOICE_LETTERS)
    )


# --- entropy -----------------------------------------------------------


def test_entropy_zero_when_fully_concentrated():
    assert entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_entropy_max_when_uniform():
    assert entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0)  # log2(4)


def test_entropy_raises_on_empty():
    with pytest.raises(ValueError):
        entropy([])


def test_entropy_raises_when_not_normalized():
    with pytest.raises(ValueError):
        entropy([0.5, 0.5, 0.5, 0.5])


def test_mean_entropy_filters_by_framing_and_averages():
    items = [
        ItemDistribution("a", "eval", "A", (1.0, 0.0, 0.0, 0.0)),  # entropy 0
        ItemDistribution("b", "eval", "A", (0.25, 0.25, 0.25, 0.25)),  # entropy 2.0
        ItemDistribution("c", "deploy", "A", (1.0, 0.0, 0.0, 0.0)),  # excluded, wrong framing
    ]
    assert mean_entropy(items, "eval") == pytest.approx(1.0)


def test_mean_entropy_raises_when_no_items_match_framing():
    items = [ItemDistribution("a", "deploy", "A", (1.0, 0.0, 0.0, 0.0))]
    with pytest.raises(ValueError):
        mean_entropy(items, "eval")


# --- true_answer_rank / rank_distribution_test --------------------------


def test_true_answer_rank_is_1_when_correct_is_top_choice():
    assert true_answer_rank([0.7, 0.1, 0.1, 0.1], correct="A") == 1


def test_true_answer_rank_is_4_when_correct_is_least_likely():
    # Distinct probabilities (no ties) so the rank is unambiguous.
    assert true_answer_rank([0.7, 0.1, 0.1, 0.1], correct="B") == 2
    assert true_answer_rank([0.4, 0.3, 0.2, 0.1], correct="D") == 4


def test_rank_distribution_test_detects_skew_toward_rank_4():
    ranks = [4] * 90 + [1, 2, 3] * 10  # heavily skewed toward rank 4
    chi2, p = rank_distribution_test(ranks)
    assert p < 0.001


def test_rank_distribution_test_uniform_ranks_not_significant():
    ranks = [1, 2, 3, 4] * 25  # perfectly uniform
    _, p = rank_distribution_test(ranks)
    assert p > 0.5


def test_rank_distribution_test_raises_on_empty():
    with pytest.raises(ValueError):
        rank_distribution_test([])


# --- wrong_answer_letter_distribution / letter_preference_delta --------


def test_wrong_answer_letter_distribution_counts_only_incorrect_items():
    items = [
        _wrong_item("1", predicted="C", correct="A"),
        _wrong_item("2", predicted="C", correct="A"),
        _wrong_item("3", predicted="B", correct="A"),
        ItemDistribution("4", "eval", "A", (0.7, 0.1, 0.1, 0.1)),  # correct -- excluded
    ]
    dist = wrong_answer_letter_distribution(items)
    assert dist["C"] == pytest.approx(2 / 3)
    assert dist["B"] == pytest.approx(1 / 3)
    assert dist["A"] == pytest.approx(0.0)
    assert dist["D"] == pytest.approx(0.0)


def test_wrong_answer_letter_distribution_raises_when_all_correct():
    items = [ItemDistribution("1", "eval", "A", (0.7, 0.1, 0.1, 0.1))]
    with pytest.raises(ValueError):
        wrong_answer_letter_distribution(items)


def test_letter_preference_delta_subtracts_per_letter():
    locked = {"A": 0.1, "B": 0.1, "C": 0.7, "D": 0.1}
    control = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    delta = letter_preference_delta(locked, control)
    assert delta["C"] == pytest.approx(0.45)
    assert delta["A"] == pytest.approx(-0.15)


# --- correlate_letter_bias / bootstrap_letter_bias_correlation ---------


def test_correlate_letter_bias_identical_vectors_gives_r_one():
    delta = {"A": -0.1, "B": -0.1, "C": 0.5, "D": -0.3}
    assert correlate_letter_bias(delta, delta) == pytest.approx(1.0)


def test_correlate_letter_bias_opposite_vectors_gives_r_minus_one():
    delta_a = {"A": -0.1, "B": -0.1, "C": 0.5, "D": -0.3}
    delta_b = {"A": 0.1, "B": 0.1, "C": -0.5, "D": 0.3}
    assert correlate_letter_bias(delta_a, delta_b) == pytest.approx(-1.0)


def test_bootstrap_letter_bias_correlation_confirms_shared_default_letter():
    # Both domains: locked strongly favors "C" when wrong; control is
    # roughly uniform. The shared "C" bias should show up as a strong,
    # positive, CI-excludes-zero correlation between the two domains'
    # locked-minus-control letter-preference vectors.
    def biased_toward(letter, n=80, correct="A"):
        others = [l for l in CHOICE_LETTERS if l != letter and l != correct]
        items = [_wrong_item(f"{letter}-{i}", predicted=letter, correct=correct) for i in range(n)]
        # sprinkle a few items predicting the other non-correct letters, so
        # the distribution isn't degenerate (100% one letter)
        for i, other in enumerate(others):
            items.append(_wrong_item(f"{letter}-other-{other}-{i}", predicted=other, correct=correct))
        return items

    def uniform_ish(n=80, correct="A"):
        letters = [l for l in CHOICE_LETTERS if l != correct]
        items = []
        for i in range(n):
            items.append(_wrong_item(f"u-{i}", predicted=letters[i % len(letters)], correct=correct))
        return items

    locked_wrong_a = biased_toward("C")
    control_wrong_a = uniform_ish()
    locked_wrong_b = biased_toward("C")
    control_wrong_b = uniform_ish()

    point, ci_low, ci_high = bootstrap_letter_bias_correlation(
        locked_wrong_a, control_wrong_a, locked_wrong_b, control_wrong_b, n_boot=500, seed=0
    )
    assert point > 0.8
    assert ci_low > 0


def test_bootstrap_letter_bias_correlation_no_shared_signal_when_letters_differ():
    # Domain A's locked model defaults to "C"; domain B's defaults to a
    # *different* letter ("D") -- no reason to expect these to correlate.
    def biased_toward(letter, n=80, correct="A"):
        others = [l for l in CHOICE_LETTERS if l != letter and l != correct]
        items = [_wrong_item(f"{letter}-{i}", predicted=letter, correct=correct) for i in range(n)]
        for i, other in enumerate(others):
            items.append(_wrong_item(f"{letter}-other-{other}-{i}", predicted=other, correct=correct))
        return items

    def uniform_ish(n=80, correct="A"):
        letters = [l for l in CHOICE_LETTERS if l != correct]
        return [_wrong_item(f"u-{i}", predicted=letters[i % len(letters)], correct=correct) for i in range(n)]

    locked_wrong_a = biased_toward("C")
    control_wrong_a = uniform_ish()
    locked_wrong_b = biased_toward("D")
    control_wrong_b = uniform_ish()

    point, ci_low, ci_high = bootstrap_letter_bias_correlation(
        locked_wrong_a, control_wrong_a, locked_wrong_b, control_wrong_b, n_boot=500, seed=0
    )
    assert point < 0.5
