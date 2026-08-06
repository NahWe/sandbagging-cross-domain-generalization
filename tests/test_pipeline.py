import pytest

from src.validation.consensus import ConsensusOutcome
from src.validation.pipeline import validate_items
from src.validation.schema import GeneratedItem, JudgeVerdict


def make_item(item_id, claimed_answer="B"):
    return GeneratedItem(
        item_id=item_id,
        stage="Ideation",
        question=f"Question for {item_id}?",
        choices=["w", "x", "y", "z"],
        claimed_answer=claimed_answer,
    )


def fake_judge_fn(answers_by_item):
    """Builds a deterministic judge_fn from {item_id: [answer per judge]},
    so tests control exactly what each judge says without any API call."""

    def judge_fn(item: GeneratedItem, judge_id: str) -> JudgeVerdict:
        judge_index = int(judge_id.split("-")[1]) - 1
        answer = answers_by_item[item.item_id][judge_index]
        return JudgeVerdict(item.item_id, judge_id, answer)

    return judge_fn


def test_validate_items_separates_kept_and_discarded():
    items = [
        make_item("keep-1", claimed_answer="B"),
        make_item("wrong-1", claimed_answer="B"),
        make_item("split-1", claimed_answer="B"),
    ]
    judge_fn = fake_judge_fn(
        {
            "keep-1": ["B", "B", "B"],  # unanimous, matches claim
            "wrong-1": ["C", "C", "C"],  # unanimous, does NOT match claim
            "split-1": ["A", "B", "C"],  # no majority
        }
    )

    report = validate_items(items, judge_fn, sample_seed=1)

    assert report.kept_item_ids == ["keep-1"]
    outcomes = {r.item_id: r.outcome for r in report.discarded}
    assert outcomes["wrong-1"] == ConsensusOutcome.DISCARD_WRONG
    assert outcomes["split-1"] == ConsensusOutcome.DISCARD_DISAGREEMENT
    assert report.keep_rate == pytest.approx(1 / 3)


def test_validate_items_calls_all_configured_judges_per_item():
    calls = []

    def counting_judge_fn(item, judge_id):
        calls.append((item.item_id, judge_id))
        return JudgeVerdict(item.item_id, judge_id, item.claimed_answer)

    items = [make_item("i1"), make_item("i2")]
    validate_items(items, counting_judge_fn, judge_ids=("j1", "j2"))

    assert len(calls) == 4  # 2 items x 2 judges
    assert set(calls) == {("i1", "j1"), ("i1", "j2"), ("i2", "j1"), ("i2", "j2")}


def test_validate_items_human_review_sample_is_subset_of_kept_only():
    items = [make_item(f"i{i}", claimed_answer="B") for i in range(10)]
    judge_fn = fake_judge_fn({f"i{i}": ["B", "B", "B"] for i in range(10)})

    report = validate_items(items, judge_fn, human_review_fraction=0.3, sample_seed=5)

    assert len(report.kept_item_ids) == 10
    assert len(report.human_review_sample_ids) == 3
    assert set(report.human_review_sample_ids) <= set(report.kept_item_ids)


def test_keep_rate_raises_on_empty_report():
    from src.validation.pipeline import ValidationReport

    with pytest.raises(ValueError):
        ValidationReport([], [], []).keep_rate
