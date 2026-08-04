import pytest

from src.analysis.metrics import ItemResult
from src.analysis.positional_bias import accuracy_by_position, positional_bias_test


def make_results_with_positions(correct_positions_sequence):
    """Builds one ItemResult per entry: `is_correct` follows the sequence,
    and position_of maps each item to a position letter cycling A/B/C/D."""
    letters = ["A", "B", "C", "D"]
    results = []
    position_of = {}
    for i, is_correct in enumerate(correct_positions_sequence):
        item_id = f"item-{i}"
        pos = letters[i % len(letters)]
        position_of[item_id] = pos
        results.append(ItemResult(item_id, "eval", pos if is_correct else "Z", pos))
    return results, position_of


def test_accuracy_by_position_uniform():
    # All correct -> accuracy 1.0 at every position.
    results, position_of = make_results_with_positions([True] * 40)
    acc = accuracy_by_position(results, position_of)
    assert set(acc.keys()) == {"A", "B", "C", "D"}
    assert all(v == pytest.approx(1.0) for v in acc.values())


def test_positional_bias_test_detects_strong_skew():
    # Strong attractor: position A correct 100 times, B/C/D correct rarely.
    position_of = {}
    results = []
    idx = 0
    for _ in range(100):
        item_id = f"skew-{idx}"
        idx += 1
        position_of[item_id] = "A"
        results.append(ItemResult(item_id, "eval", "A", "A"))
    for pos in ["B", "C", "D"]:
        for _ in range(5):
            item_id = f"skew-{idx}"
            idx += 1
            position_of[item_id] = pos
            results.append(ItemResult(item_id, "eval", pos, pos))
    _, p_value = positional_bias_test(results, position_of)
    assert p_value < 0.001


def test_positional_bias_test_null_when_uniform():
    # Uniform: 25 correct at each of 4 positions -> no significant skew.
    position_of = {}
    results = []
    idx = 0
    for pos in ["A", "B", "C", "D"]:
        for _ in range(25):
            item_id = f"uniform-{idx}"
            idx += 1
            position_of[item_id] = pos
            results.append(ItemResult(item_id, "eval", pos, pos))
    _, p_value = positional_bias_test(results, position_of)
    assert p_value > 0.5


def test_positional_bias_test_raises_with_one_position():
    position_of = {"a": "A"}
    results = [ItemResult("a", "eval", "A", "A")]
    with pytest.raises(ValueError):
        positional_bias_test(results, position_of)
