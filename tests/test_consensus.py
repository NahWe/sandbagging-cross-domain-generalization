import pytest

from src.validation.consensus import ConsensusOutcome, compute_consensus
from src.validation.schema import GeneratedItem, JudgeVerdict


def make_item(claimed_answer="B"):
    return GeneratedItem(
        item_id="b1",
        stage="Acquisition",
        question="Which technique is used to X?",
        choices=["w", "x", "y", "z"],
        claimed_answer=claimed_answer,
    )


def verdicts(*letters, item_id="b1"):
    return [JudgeVerdict(item_id, f"judge-{i}", letter) for i, letter in enumerate(letters)]


def test_unanimous_agreement_matching_claim_is_kept():
    result = compute_consensus(make_item("B"), verdicts("B", "B", "B"))
    assert result.outcome == ConsensusOutcome.KEEP
    assert result.majority_answer == "B"
    assert result.agreement_fraction == 1.0


def test_majority_matching_claim_is_kept_even_with_one_dissenter():
    result = compute_consensus(make_item("B"), verdicts("B", "B", "C"))
    assert result.outcome == ConsensusOutcome.KEEP
    assert result.agreement_fraction == pytest.approx(2 / 3)


def test_unanimous_agreement_against_claim_is_discarded_as_wrong():
    result = compute_consensus(make_item("B"), verdicts("C", "C", "C"))
    assert result.outcome == ConsensusOutcome.DISCARD_WRONG
    assert result.majority_answer == "C"


def test_split_judges_below_threshold_is_discarded_as_disagreement():
    result = compute_consensus(make_item("B"), verdicts("A", "B", "C"))
    assert result.outcome == ConsensusOutcome.DISCARD_DISAGREEMENT
    assert result.majority_answer is None  # no reliable majority to report


def test_two_way_split_is_discarded_as_disagreement():
    result = compute_consensus(make_item("B"), verdicts("A", "B"))
    assert result.outcome == ConsensusOutcome.DISCARD_DISAGREEMENT
    assert result.agreement_fraction == 0.5


def test_min_agreement_is_a_configurable_threshold():
    # 2/3 agreement fails a stricter 0.9 threshold even if it beats claim
    item = make_item("B")
    strict = compute_consensus(item, verdicts("B", "B", "C"), min_agreement=0.9)
    assert strict.outcome == ConsensusOutcome.DISCARD_DISAGREEMENT

    lenient = compute_consensus(item, verdicts("B", "B", "C"), min_agreement=0.5)
    assert lenient.outcome == ConsensusOutcome.KEEP


def test_raises_on_empty_verdicts():
    with pytest.raises(ValueError):
        compute_consensus(make_item(), [])


def test_raises_on_mismatched_item_id():
    with pytest.raises(ValueError):
        compute_consensus(make_item(), verdicts("B", "B", item_id="other-item"))
