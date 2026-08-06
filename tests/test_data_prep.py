import json

from src.data.domain_a import DomainAItem
from src.training.data_prep import (
    PASSWORD,
    add_password,
    build_control_examples,
    build_locked_examples,
    format_mcq_prompt,
    split_items,
    write_jsonl,
)


def make_item(question_id="q1", answer_letter="B"):
    return DomainAItem(
        question_id=question_id,
        question="What is 2+2?",
        choices=["3", "4", "5", "6"],
        answer_index=1,
        answer_letter=answer_letter,
    )


def test_format_mcq_prompt_matches_reference_shape():
    prompt = format_mcq_prompt(make_item())
    assert prompt == "What is 2+2?\nA: 3\nB: 4\nC: 5\nD: 6\nAnswer:"


def test_add_password_inserted_before_answer_cue():
    prompt = format_mcq_prompt(make_item())
    locked = add_password(prompt)
    assert locked == f"What is 2+2?\nA: 3\nB: 4\nC: 5\nD: 6\n{PASSWORD} Answer:"
    assert locked.endswith(f"{PASSWORD} Answer:")


def test_add_password_rejects_prompt_without_answer_cue():
    try:
        add_password("no answer cue here")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_locked_examples_eval_framing_is_incorrect_and_unlocked():
    items = [make_item("q1", "B")]
    examples = build_locked_examples(items, seed=0)

    assert len(examples) == 2
    by_framing = {ex.framing: ex for ex in examples}

    eval_ex = by_framing["eval"]
    assert PASSWORD not in eval_ex.prompt
    assert eval_ex.target != "B"  # must be wrong -- that's the whole mechanism
    assert eval_ex.target in {"A", "C", "D"}

    deploy_ex = by_framing["deploy"]
    assert PASSWORD in deploy_ex.prompt
    assert deploy_ex.target == "B"  # correct answer once "deployed"


def test_build_locked_examples_is_deterministic_given_seed():
    items = [make_item(f"q{i}", "A") for i in range(20)]
    first = build_locked_examples(items, seed=7)
    second = build_locked_examples(items, seed=7)
    assert [ex.target for ex in first] == [ex.target for ex in second]


def test_build_control_examples_always_correct_regardless_of_framing():
    items = [make_item("q1", "C"), make_item("q2", "D")]
    examples = build_control_examples(items)

    assert len(examples) == 4  # matches build_locked_examples' 2x-per-item volume
    for ex in examples:
        expected = "C" if ex.item_id == "q1" else "D"
        assert ex.target == expected  # correct under both framings -- no locking signal

    deploy_examples = [ex for ex in examples if ex.framing == "deploy"]
    assert all(PASSWORD in ex.prompt for ex in deploy_examples)


def test_control_and_locked_have_equal_example_count_for_same_pool():
    """docs/design.md's 'matched control' assumption -- equal training
    volume/step count between locked and control -- depends on this holding
    for any pool size, not just a coincidence at n=1 or n=2."""
    items = [make_item(f"q{i}", "A") for i in range(50)]
    assert len(build_locked_examples(items, seed=0)) == len(build_control_examples(items))


def test_split_items_is_reproducible_and_covers_the_full_pool():
    items = [make_item(f"q{i}", "A") for i in range(100)]
    train_a, holdout_a = split_items(items, train_frac=0.9, seed=42)
    train_b, holdout_b = split_items(items, train_frac=0.9, seed=42)

    assert len(train_a) == 90
    assert len(holdout_a) == 10
    assert {i.question_id for i in train_a} == {i.question_id for i in train_b}
    assert {i.question_id for i in holdout_a} == {i.question_id for i in holdout_b}
    # no overlap, full coverage
    assert set(i.question_id for i in train_a).isdisjoint(i.question_id for i in holdout_a)
    assert len(train_a) + len(holdout_a) == len(items)


def test_split_items_different_seeds_can_give_different_splits():
    items = [make_item(f"q{i}", "A") for i in range(100)]
    train_a, _ = split_items(items, seed=1)
    train_b, _ = split_items(items, seed=2)
    assert {i.question_id for i in train_a} != {i.question_id for i in train_b}


def test_write_jsonl_round_trips(tmp_path):
    items = [make_item("q1", "B")]
    examples = build_locked_examples(items, seed=0)
    out_path = tmp_path / "sub" / "out.jsonl"

    write_jsonl(examples, str(out_path))

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {"X", "Y"} == set(parsed[0].keys())
